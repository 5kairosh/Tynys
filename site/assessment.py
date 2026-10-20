"""Conservative display guard; not a validated selective classifier.

The existing OOD flag compares features with the validation distribution. It
neither authenticates X-rays nor estimates clinical reliability. Keep raw model
classes/scores available for inspection, but withhold a headline classification
when the distribution check flags the input or cannot be completed.
"""

POLICY = 'ood_guard_v2'
REVIEW_TITLE = 'Сенімді бағалау мүмкін емес'
OOD_SHIFT = ('Сурет модель тексерілген деректерден ерекшеленеді (OOD). '
             'Рентген суреттерінде де осындай ескерту шығуы мүмкін. '
             'Санат бойынша қорытынды берілмейді; төмендегі баллдар тек техникалық нәтиже.')
OOD_UNAVAILABLE = ('OOD тексеруі аяқталмады немесе қолжетімсіз. '
                   'Санат бойынша қорытынды берілмейді; төмендегі баллдар тек техникалық нәтиже.')
LIMIT = ('Бұл — OOD-қа сүйенетін сақтық ережесі. '
         'Ішкі және шағын сыртқы жиындағы өлшемдер клиникалық сенімділікті растамайды.')


def assess_input(check):
    check = check if isinstance(check, dict) else {}
    complete = check.get('available') is True and type(check.get('is_xray')) is bool
    allowed = complete and check['is_xray'] is True
    return {
        'status': 'classified' if allowed else 'review_required',
        'reason': 'within_reference' if allowed else 'ood_shift' if complete else 'ood_unavailable',
        'title': 'Модель классификациясы' if allowed else REVIEW_TITLE,
        'message': ('OOD бойынша train үлестірімінен айқын ауытқу табылмады; шек бөлек validation-да бапталған. '
                    'Бұл рентгеннің жарамдылығын немесе болжамның дұрыстығын растамайды.')
                   if allowed else OOD_SHIFT if complete else OOD_UNAVAILABLE,
        'policy': POLICY,
        'clinically_validated': False,
        'notice': LIMIT,
    }
