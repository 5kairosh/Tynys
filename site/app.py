"""Tynys: the original Colab preprocessing and Keras model, behind a local API."""
import csv
import base64
import hashlib
import io
import json
import logging
import math
import threading
import time
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
import numpy as np
from starlette.concurrency import run_in_threadpool

from calibration import calibrate_score, validate_calibration
from ood import OODDetector
from dicom_input import dicom_to_png, dicom_status
from pad_geometry import pad_geometry

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
IMAGE_LIMIT = 12 * 1024 * 1024
MODEL_LIMIT = 150 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = 20_000_000


def read_config(path):
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("classes") != ["NORMAL", "PNEUMONIA"]:
        raise ValueError("Модель санаттары NORMAL және PNEUMONIA болуы керек.")
    size = config.get("image_size")
    if not isinstance(size, list) or len(size) != 2 or any(type(n) is not int or n < 32 or n > 1024 for n in size):
        raise ValueError("Сурет өлшемі конфигурацияда дұрыс берілмеген.")
    threshold = config.get("threshold")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Шешім шегі 0 мен 1 арасында болуы керек.")
    validate_calibration(config.get("calibration"))
    for value in config.get("model_calibrations", {}).values():
        validate_calibration(value)
    return config


class Engine:
    def __init__(self):
        self.lock = threading.Lock()
        self.model = None
        self.tf = None
        self.config = None
        self.metrics = []
        self.error = None
        self.fingerprint = None
        self.cnn_model = None
        self.cnn_threshold = None
        self.cnn_fingerprint = None
        self.comparison_error = None
        self.ood = OODDetector()

    def load(self):
        with self.lock:
            try:
                import tensorflow as tf
                config = read_config(MODEL_DIR / "model_config.json")
                model = tf.keras.models.load_model(MODEL_DIR / "pneumonia_mobilenetv2.keras", compile=False, safe_mode=True)
                if tuple(model.input_shape[1:]) != (*config["image_size"], 3) or tuple(model.output_shape[1:]) != (1,):
                    raise ValueError("Модельдің кірісі не шығысы конфигурацияға сәйкес емес.")
                # Warm the actual model before advertising readiness.
                model(tf.zeros([1, *config["image_size"], 3]), training=False)
                metrics_path = MODEL_DIR / "test_metrics.csv"
                metrics = []
                if metrics_path.exists():
                    with metrics_path.open(encoding="utf-8", newline="") as stream:
                        for row in csv.DictReader(stream):
                            metrics.append({k: (v if k == "model" else float(v)) for k, v in row.items()})
                self.tf, self.model, self.config, self.metrics = tf, model, config, metrics
                self.fingerprint = hashlib.sha256((MODEL_DIR / "pneumonia_mobilenetv2.keras").read_bytes()).hexdigest()[:12]
                self.error = None
                self.ood.load(MODEL_DIR/"ood_params.npz", self.fingerprint)
                try:
                    cnn_path = MODEL_DIR/'cnn_best.keras'
                    cnn_rows = [r for r in metrics if r['model'] == 'CNN']
                    if len(cnn_rows) != 1:
                        raise ValueError('Exactly one CNN threshold is required from test_metrics.csv.')
                    cnn_threshold = cnn_rows[0]['threshold']
                    if not math.isfinite(cnn_threshold) or not 0 <= cnn_threshold <= 1:
                        raise ValueError('CNN threshold is invalid.')
                    cnn_model = tf.keras.models.load_model(cnn_path, compile=False, safe_mode=True)
                    if tuple(cnn_model.input_shape[1:]) != (*config['image_size'],3) or tuple(cnn_model.output_shape[1:]) != (1,):
                        raise ValueError('CNN input/output shape does not match image pipeline.')
                    cnn_model(tf.zeros([1,*config['image_size'],3]),training=False)
                    self.cnn_model, self.cnn_threshold = cnn_model, cnn_threshold
                    self.cnn_fingerprint = hashlib.sha256(cnn_path.read_bytes()).hexdigest()[:12]
                    self.comparison_error = None
                except Exception:
                    logging.exception('CNN comparison model unavailable')
                    self.comparison_error = 'CNN моделі немесе оның сақталған шегі жүктелмеді. Екі модельді салыстыру қолжетімсіз.'
            except Exception:
                logging.exception("Cannot load trained model")
                self.error = "Модель жүктелмеді. Сервер журналын және model бумасын тексер."

    def paired_test_status(self):
        try:
            value = json.loads((MODEL_DIR/'mcnemar.json').read_text(encoding='utf-8'))
            ids = {'CNN':self.cnn_fingerprint,'MobileNetV2':self.fingerprint}
            config_hash = hashlib.sha256((MODEL_DIR/'model_config.json').read_bytes()).hexdigest()
            if value['model_ids'] != ids or value['config_sha256'] != config_hash:
                raise ValueError('Артефакт басқа модельге не конфигурацияға арналған.')
            return value
        except Exception:
            return {'available':False,'message':'Осы модельдерге сәйкес McNemar нәтижесі жоқ.'}

    def status(self):
        return {"dicom": dicom_status(), "xray_check": self.ood.status(), "mcnemar": self.paired_test_status(), "ready": self.model is not None, "message": self.error,
                "config": self.config, "metrics": self.metrics,
                "model_id": self.fingerprint,
                "comparison_ready": self.cnn_model is not None,
                "cnn_threshold": self.cnn_threshold,
                "comparison_message": self.comparison_error,
                "tensorflow_version": self.tf.__version__ if self.tf else None,
                "calibration": self.config.get("calibration") if self.config else None}

    def gradcam(self, image, original_rgb, positive):
        """Class-specific Grad-CAM on MobileNetV2's last spatial feature map.

        Use the pre-sigmoid logit to avoid sigmoid saturation. NORMAL targets
        the negative logit. Never mutate the model's activation or weights.
        """
        tf = self.tf
        layers = self.model.layers
        base_index = next(i for i, layer in enumerate(layers)
                          if isinstance(layer, tf.keras.Model)
                          and layer.name.startswith('mobilenetv2')
                          and len(layer.output_shape) == 4)
        x = image[None, ...]
        for layer in layers[1:base_index]:
            x = layer(x, training=False)
        feature_maps = layers[base_index](x, training=False)
        dense = layers[-1]
        if not isinstance(dense, tf.keras.layers.Dense) or dense.units != 1:
            raise ValueError('Grad-CAM expects the trained one-unit sigmoid head.')
        with tf.GradientTape() as tape:
            tape.watch(feature_maps)
            head = feature_maps
            for layer in layers[base_index + 1:-1]:
                head = layer(head, training=False)
            logits = tf.linalg.matmul(head, dense.kernel)
            if dense.use_bias:
                logits = logits + dense.bias
            target = logits[:, 0] if positive else -logits[:, 0]
        gradients = tape.gradient(target, feature_maps)
        if gradients is None:
            raise ValueError('Grad-CAM gradient is disconnected.')
        weights = tf.reduce_mean(gradients, axis=(0, 1, 2))
        cam = tf.nn.relu(tf.reduce_sum(feature_maps[0] * weights, axis=-1))
        maximum = float(tf.reduce_max(cam))
        if not math.isfinite(maximum) or maximum <= 0:
            return {"available": False, "message": "Осы суретте оң Grad-CAM белсенділігі табылмады. Аймақ белгіленбейді."}
        cam = cam / maximum
        target_h, target_w = self.config['image_size']
        cam = tf.image.resize(cam[..., None], [target_h, target_w])[..., 0]
        source_h, source_w = original_rgb.shape[:2]
        # Invert resize_with_pad's exact float32 resize/crop geometry.
        geometry = pad_geometry(source_h,source_w,target_h,target_w)
        resized_h,resized_w = geometry['resized_h'],geometry['resized_w']
        pad_y,pad_x = geometry['pad_y'],geometry['pad_x']
        cam = cam[pad_y:pad_y + resized_h, pad_x:pad_x + resized_w]
        original = Image.fromarray(original_rgb.astype(np.uint8), 'RGB')
        original.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
        heat = tf.image.resize(cam[..., None], [original.height, original.width])[..., 0].numpy()
        heat = np.clip(heat, 0, 1)
        # Transparent at weak activation, amber/red at stronger activation.
        color = np.stack([np.full_like(heat, 255), 215 * (1 - heat),
                          60 * (1 - heat)], axis=-1)
        alpha = (0.62 * np.clip((heat - 0.15) / 0.85, 0, 1))[..., None]
        overlay = np.clip(np.asarray(original) * (1 - alpha) + color * alpha, 0, 255).astype(np.uint8)
        output = io.BytesIO()
        Image.fromarray(overlay).save(output, format='PNG')
        return {"available": True, "method": "Grad-CAM", "target_class": 'PNEUMONIA' if positive else 'NORMAL',
                "overlay": 'data:image/png;base64,' + base64.b64encode(output.getvalue()).decode('ascii'),
                "feature_map_size": list(feature_maps.shape[1:3]),
                "reconstructed_score": float(tf.sigmoid(logits)[0, 0]),
                "notice": "Қызыл-сары аймақтар таңдалған санатқа оң ықпал еткен модель белсенділігін көрсетеді. Бұл ауру ошағы немесе оның шекарасы емес."}

    def predict(self, data, filename):
        if self.model is None:
            raise HTTPException(503, self.error or "Модель әлі дайын емес.")
        is_dicom = Path(filename).suffix.lower() == '.dcm'
        if is_dicom:
            data = dicom_to_png(data)
        try:
            with Image.open(io.BytesIO(data)) as original:
                if original.format not in {"JPEG", "PNG"}:
                    raise ValueError("JPG немесе PNG суретін таңда.")
                if original.width * original.height > 20_000_000:
                    raise ValueError("Сурет тым үлкен. 20 мегапиксельге дейінгі файл таңда.")
                width, height = original.size
                original.verify()
        except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
            raise HTTPException(400, "Суретті оқу мүмкін емес. Дұрыс JPG немесе PNG файлын таңда.") from exc
        started = time.perf_counter()
        try:
            with self.lock:
                tf = self.tf
                # Exactly the notebook's decode_image: no EXIF rotation, no extra
                # normalization. MobileNet Rescaling is already inside the model.
                original_image = tf.io.decode_image(data, channels=3, expand_animations=False)
                original_image.set_shape([None, None, 3])
                image = tf.image.resize_with_pad(tf.cast(original_image, tf.float32), *self.config["image_size"])
                raw_score = float(self.model(image[None, ...], training=False).numpy()[0, 0])
                score = calibrate_score(raw_score, self.config.get("calibration"))
                threshold = float(self.config["threshold"])
                xray_check = self.ood.status()
                if xray_check['available']:
                    try:
                        x = image[None,...]
                        for layer in self.model.layers[1:]:
                            x = layer(x,training=False)
                            if isinstance(layer,tf.keras.Model) and layer.name.startswith('mobilenetv2'):
                                break
                        vector = tf.reduce_mean(x,axis=(1,2)).numpy()[0]
                        xray_check = self.ood.check(vector)
                    except Exception:
                        logging.exception('OOD warning unavailable')
                        xray_check = {'available':False,'message':'OOD қашықтығын есептеу мүмкін болмады.'}
                comparison = {'available': False, 'message': self.comparison_error}
                if self.cnn_model is not None:
                    cnn_score = float(self.cnn_model(image[None,...],training=False).numpy()[0,0])
                    cnn_score = calibrate_score(cnn_score, self.config.get('model_calibrations', {}).get('CNN'))
                    if not math.isfinite(cnn_score) or not 0 <= cnn_score <= 1:
                        raise ValueError('CNN returned an invalid score.')
                    cnn_class = self.config['classes'][int(cnn_score >= self.cnn_threshold)]
                    mobile_class = self.config['classes'][int(score >= threshold)]
                    comparison = {'available': True, 'agreement': cnn_class == mobile_class,
                                  'models': [
                                      {'name': 'CNN', 'model_class': cnn_class, 'pneumonia_score': cnn_score,
                                       'threshold': self.cnn_threshold, 'model_id': self.cnn_fingerprint},
                                      {'name': 'MobileNetV2', 'model_class': mobile_class, 'pneumonia_score': score,
                                       'threshold': threshold, 'model_id': self.fingerprint}]}
                try:
                    explanation = self.gradcam(image, original_image.numpy(), score >= threshold)
                    if explanation.get('available') and abs(explanation['reconstructed_score'] - raw_score) > 1e-5:
                        raise ValueError('Grad-CAM head does not match model prediction.')
                except Exception:
                    logging.exception('Grad-CAM unavailable; prediction remains valid')
                    explanation = {"available": False, "message": "Жылу картасын есептеу мүмкін болмады. Төменде модельдің жіктеу нәтижесі көрсетілген."}
        except Exception as exc:
            logging.exception("Image inference failed")
            raise HTTPException(422, "Суретті модельмен өңдеу мүмкін болмады.") from exc
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise HTTPException(500, "Модель жарамсыз балл берді.")
        return {"model_class": self.config["classes"][int(score >= threshold)],
                "input_format": "DICOM" if is_dicom else "image",
                "preview": "data:image/png;base64,"+base64.b64encode(data).decode("ascii") if is_dicom else None,
                "pneumonia_score": score, "raw_score": raw_score, "calibration": self.config.get("calibration"), "threshold": threshold,
                "filename": Path(filename).name, "width": width, "height": height,
                "elapsed_ms": round((time.perf_counter() - started) * 1000),
                "model_id": self.fingerprint,
                "explanation": explanation, "xray_check": xray_check,
                "comparison": comparison,
                "notice": "Оқу моделінің болжамы. Медициналық диагноз емес."}


engine = Engine()


@asynccontextmanager
async def lifespan(app):
    await run_in_threadpool(engine.load)
    yield


app = FastAPI(title="Tynys", lifespan=lifespan)


@app.middleware("http")
async def privacy_headers(request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response


@app.get("/api/status")
def status():
    return engine.status()


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    if Path(file.filename or "").suffix.lower() not in {".jpg", ".jpeg", ".png", ".dcm"}:
        raise HTTPException(400, "JPG, PNG немесе DICOM файлын таңда.")
    data = await file.read(IMAGE_LIMIT + 1)
    await file.close()
    if not data:
        raise HTTPException(400, "Файл бос.")
    if len(data) > IMAGE_LIMIT:
        raise HTTPException(413, "Файл көлемі 12 МБ-тан аспауы керек.")
    return await run_in_threadpool(engine.predict, data, file.filename or "xray.png")


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")


@app.post('/api/report')
async def report(files: list[UploadFile] = File(...)):
    if not 1 <= len(files) <= 10:
        for file in files:
            await file.close()
        raise HTTPException(400, 'PDF үшін 1-ден 10-ға дейін сурет таңда.')
    inputs = []
    try:
        for file in files:
            if Path(file.filename or '').suffix.lower() not in {'.jpg', '.jpeg', '.png', '.dcm'}:
                raise HTTPException(400, 'PDF үшін JPG, PNG немесе DICOM суретін таңда.')
            data = await file.read(IMAGE_LIMIT + 1)
            if not data or len(data) > IMAGE_LIMIT:
                raise HTTPException(400 if not data else 413, 'Әр файл бос болмауы және 12 МБ-тан аспауы керек.')
            inputs.append((data, file.filename or 'xray.png'))
    finally:
        for file in files:
            await file.close()

    def generate():
        from pdf_report import make_report
        items = []
        for data,name in inputs:
            result = engine.predict(data,name)
            display = base64.b64decode(result['preview'].split(',',1)[1]) if result.get('preview') else data
            items.append((display,result))
        return make_report(items)
    pdf = await run_in_threadpool(generate)
    return Response(pdf, media_type='application/pdf',
                    headers={'Content-Disposition': 'attachment; filename="tynys-report.pdf"'})


app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=7860)
