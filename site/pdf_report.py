"""Generate a Kazakh PDF from actual inference results, entirely in memory."""
import base64
import io
from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import escape

from PIL import Image
from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph

ROOT = Path(__file__).resolve().parent
pdfmetrics.registerFont(TTFont('TynysSans', str(ROOT/'fonts/NotoSans-Regular.ttf')))
pdfmetrics.registerFont(TTFont('TynysBold', str(ROOT/'fonts/NotoSans-Bold.ttf')))
INK = HexColor('#193c37')
GREEN = HexColor('#08796e')
MUTED = HexColor('#647b75')
LINE = HexColor('#dfebe5')


def decimal(number, digits=3):
    return f'{number:.{digits}f}'.replace('.', ',')


def paragraph(c, text, x, top, width, size=9, color=INK, bold=False):
    style = ParagraphStyle('body', fontName='TynysBold' if bold else 'TynysSans',
                           fontSize=size, leading=size*1.5, textColor=color)
    p = Paragraph(escape(text), style)
    _, height = p.wrap(width, 1000)
    p.drawOn(c, x, top-height)
    return top-height


def fitted_image(c, image, x, y, width, height):
    source = Image.open(io.BytesIO(image)).convert('RGB')
    source.thumbnail((1400, 1400), Image.Resampling.LANCZOS)
    c.drawImage(ImageReader(source), x, y, width=width, height=height,
                preserveAspectRatio=True, anchor='c', anchorAtXY=False)


def make_report(items):
    """items are (original bytes, server-computed prediction) pairs."""
    output = io.BytesIO()
    c = canvas.Canvas(output, pagesize=A4, pageCompression=1)
    c.setTitle('Tynys - Рентгенді классификациялау есебі')
    c.setAuthor('Tynys оқу жобасы')
    page_width, page_height = A4
    margin = 38
    usable = page_width - 2*margin
    # Asia/Qyzylorda is UTC+5 for this project. No host-local timezone dependence.
    from datetime import timezone, timedelta
    date = datetime.now(timezone(timedelta(hours=5))).strftime('%d.%m.%Y %H:%M')
    for index, (original, r) in enumerate(items, 1):
        positive = r['model_class'] == 'PNEUMONIA'
        c.setFillColor(GREEN)
        c.setFont('TynysBold', 27)
        c.drawString(margin, page_height-58, 'tynys.')
        c.setFont('TynysSans', 10)
        c.drawRightString(page_width-margin, page_height-48, 'Рентгенді классификациялау есебі')
        c.setFillColor(MUTED)
        c.setFont('TynysSans', 8)
        c.drawRightString(page_width-margin, page_height-66, date+' (UTC+5)')
        c.setStrokeColor(LINE)
        c.line(margin, page_height-83, page_width-margin, page_height-83)

        filename = r['filename']
        if len(filename) > 90:
            filename = filename[:87]+'...'
        paragraph(c, 'Файл: '+filename, margin, 746, usable, size=9)
        paragraph(c, 'Модель санаты', margin, 713, usable, size=8, color=MUTED)
        c.setFillColor(GREEN if not positive else HexColor('#926b2c'))
        c.setFont('TynysBold', 20)
        c.drawString(margin, 674, r['model_class'])
        c.setFillColor(INK)
        c.setFont('TynysBold', 11)
        c.drawString(300, 692, 'Балл: '+decimal(r['pneumonia_score']))
        c.setFont('TynysSans', 9)
        c.drawString(300, 674, 'Шешім шегі: '+decimal(r['threshold']))
        paragraph(c, f'MobileNetV2  /  {r["width"]} × {r["height"]} px  /  модель ID: {r["model_id"]}',
                  margin, 654, usable, size=8, color=MUTED)

        gap = 16
        image_width = (usable-gap)/2
        paragraph(c, 'Түпнұсқа рентген', margin, 624, image_width, size=9, bold=True)
        explanation = r.get('explanation', {})
        paragraph(c, 'Grad-CAM: '+r['model_class'], margin+image_width+gap, 624, image_width, size=9, bold=True)
        fitted_image(c, original, margin, 424, image_width, 180)
        if explanation.get('available'):
            overlay = base64.b64decode(explanation['overlay'].split(',',1)[1])
            fitted_image(c, overlay, margin+image_width+gap, 424, image_width, 180)
        else:
            paragraph(c, explanation.get('message','Жылу картасы қолжетімсіз.'),
                      margin+image_width+gap, 505, image_width, size=9, color=MUTED)
        top = paragraph(c, 'Сары-қызыл түс таңдалған санатқа оң ықпал еткен жуық белсенділікті көрсетеді. Бұл ауру ошағы немесе оның шекарасы емес.',
                        margin, 410, usable, size=8, color=MUTED)
        check = r.get('xray_check',{})
        if check.get('available') and check.get('is_xray') is False:
            top = paragraph(c, 'Бұл сурет рентгенге ұқсамайды — болжам сенімсіз болуы мүмкін.',margin,top-4,usable,size=8,color=HexColor('#926b2c'))
        elif not check.get('available'):
            top = paragraph(c, 'OOD тексеруі қолжетімсіз.',margin,top-4,usable,size=7,color=MUTED)
        if r.get('calibration'):
            top = paragraph(c, 'Validation калибрлеуі: T='+decimal(r['calibration']['T'])+'. Клиникалық ықтималдық ретінде тексерілмеген.',margin,top-3,usable,size=7,color=MUTED)
        top = paragraph(c, 'Неге бұл санат таңдалды?', margin, top-13, usable, size=11, bold=True)
        relation = 'шекке тең немесе одан жоғары' if positive else 'шектен төмен'
        sign = '>=' if positive else '<'
        reason = (f'Модель баллы {decimal(r["pneumonia_score"],6)}, ал сақталған шек {decimal(r["threshold"],6)}. '
                  f'Балл {relation} ({decimal(r["pneumonia_score"],6)} {sign} {decimal(r["threshold"],6)}), '
                  f'сондықтан модель {r["model_class"]} санатын таңдады. Жіктеу толық дәлдіктегі баллмен жасалады.')
        top = paragraph(c, reason, margin, top-9, usable, size=9)
        comparison = r.get('comparison', {})
        if comparison.get('available'):
            top = paragraph(c, 'Осы суреттегі екі модель', margin, top-11, usable, size=10, bold=True)
            for model in comparison['models']:
                text = f'{model["name"]}: {model["model_class"]} / балл: {decimal(model["pneumonia_score"])} / шек: {decimal(model["threshold"])}'
                top = paragraph(c, text, margin, top-3, usable, size=8)
            verdict = 'Болжамдар сәйкес.' if comparison['agreement'] else 'Болжамдар әртүрлі.'
            top = paragraph(c, verdict+' Баллдың жоғары болуы болжамның дұрыстығын дәлелдемейді.', margin, top-3, usable, size=7, color=MUTED)
        top = paragraph(c, 'Нәтижені түсіндіру', margin, top-12, usable, size=11, bold=True)
        meaning = ('Модель суретті пневмония санатына жатқызды. Бұл адамның пневмониямен ауырғанын өздігінен растамайды.'
                   if positive else 'Модель баллы пневмония санатын таңдау шегіне жетпеді. Бұл пневмонияның жоқтығын немесе адамның толық сау екенін дәлелдемейді.')
        top = paragraph(c, meaning, margin, top-9, usable, size=9)
        top = paragraph(c, 'Шектеуі', margin, top-12, usable, size=10, bold=True)
        top = paragraph(c, 'Grad-CAM соңғы 7 × 7 белгілер картасынан үлкейтіледі. Карта дәл шекара, клиникалық дәлел немесе нақты медициналық белгі атауын бермейді. Балл диагноздың сенімділік пайызы емес.',
                        margin, top-8, usable, size=8, color=MUTED)
        if top < 65:
            raise ValueError('PDF content exceeded the single-page report layout.')
        c.setStrokeColor(LINE)
        c.line(margin, 47, page_width-margin, 47)
        c.setFont('TynysSans', 7.5)
        c.setFillColor(MUTED)
        c.drawString(margin, 32, 'Оқу прототипі. Медициналық диагноз емес.')
        c.drawRightString(page_width-margin, 32, f'{index} / {len(items)}')
        c.showPage()
    c.save()
    return output.getvalue()
