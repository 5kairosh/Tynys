"""Keep accepted multipart uploads in RAM and cap total request size."""
import json
import asyncio
import time
from starlette.formparsers import MultiPartParser

MAX_BODY=128*1024*1024  # Ten 12 MiB files plus multipart overhead.
PREDICT_BODY=12*1024*1024+64*1024
UPLOAD_SECONDS=60
MultiPartParser.spool_max_size=MAX_BODY+1


class UploadMemoryLimit:
    def __init__(self,app):
        self.app=app

    async def __call__(self,scope,receive,send):
        if scope['type']!='http' or scope.get('method')!='POST':
            return await self.app(scope,receive,send)
        headers=dict(scope.get('headers',[]))
        try:
            length=int(headers.get(b'content-length',b'0'))
        except ValueError:
            length=0
        async def reject(status=413):
            message=('Файлды жүктеу уақыты аяқталды. Қосылымды тексеріп, қайта басыңыз.' if status==408
                     else 'Сұрау тым үлкен. Әр файл 12 МБ-тан, сурет саны 10-нан аспауы керек.')
            data=json.dumps({'detail':message},ensure_ascii=False).encode()
            await send({'type':'http.response.start','status':status,'headers':[(b'content-type',b'application/json; charset=utf-8')]})
            await send({'type':'http.response.body','body':data})
        limit=PREDICT_BODY if scope.get('path','').rstrip('/')=='/api/predict' else MAX_BODY
        if length>limit:
            return await reject()
        # Buffer the bounded body before the multipart parser sees it. No tempfile rollover.
        body=bytearray()
        deadline=time.monotonic()+UPLOAD_SECONDS
        while True:
            try:
                event=await asyncio.wait_for(receive(), max(0,deadline-time.monotonic()))
            except TimeoutError:
                return await reject(408)
            if event['type']=='http.disconnect':
                return
            body.extend(event.get('body',b''))
            if len(body)>limit:
                return await reject()
            if not event.get('more_body',False):
                break
        delivered=False
        async def buffered_receive():
            nonlocal delivered
            if not delivered:
                delivered=True
                return {'type':'http.request','body':bytes(body),'more_body':False}
            return await receive()
        await self.app(scope,buffered_receive,send)
