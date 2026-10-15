"""Optional DICOM decoding into in-memory PNG, before the fixed TF pipeline."""
import importlib.util
import io
import numpy as np
from PIL import Image
from fastapi import HTTPException


def dicom_status():
    available = importlib.util.find_spec('pydicom') is not None
    return {'available':available,'message':None if available else 'DICOM қолжетімсіз: pydicom кітапханасы орнатылмаған.'}


def dicom_to_png(data):
    if not dicom_status()['available']:
        raise HTTPException(503,dicom_status()['message'])
    if not data:
        raise HTTPException(400,'DICOM файлы бос.')
    try:
        import pydicom
        from pydicom.pixels import apply_modality_lut
        dataset = pydicom.dcmread(io.BytesIO(data),force=False)
        if str(dataset.get('Modality','')) not in {'CR','DX'}:
            raise HTTPException(400,'Тек CR немесе DX рентген DICOM қолданылады. CT/MR қолдау көрсетілмейді.')
        rows,cols = int(dataset.Rows),int(dataset.Columns)
        if rows<=0 or cols<=0 or rows*cols>20_000_000 or int(dataset.get('NumberOfFrames',1)) != 1:
            raise HTTPException(400,'Бір кадрлы, 20 мегапиксельге дейінгі DICOM таңдаңыз.')
        if str(dataset.get('PhotometricInterpretation','')) not in {'MONOCHROME1','MONOCHROME2'}:
            raise HTTPException(400,'Тек MONOCHROME рентген DICOM қолданылады.')
        pixels = np.asarray(apply_modality_lut(dataset.pixel_array,dataset),dtype=np.float64)
        if pixels.shape != (rows,cols) or not np.isfinite(pixels).all():
            raise ValueError('Pixel array invalid')
        low,high = np.percentile(pixels,[1,99])
        if high<=low:
            raise HTTPException(400,'DICOM пиксельдерінің контрасты жоқ.')
        image = np.clip((pixels-low)/(high-low),0,1)
        if dataset.PhotometricInterpretation == 'MONOCHROME1':
            image = 1-image
        gray = (image*255).astype(np.uint8)
        rgb = np.repeat(gray[...,None],3,axis=-1)
        output = io.BytesIO()
        Image.fromarray(rgb).save(output,format='PNG')
        return output.getvalue()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(400,'DICOM файлын оқу мүмкін болмады. Файлды және сығу кодегінің қолдауын тексеріңіз.') from exc
