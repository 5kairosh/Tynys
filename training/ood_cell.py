# OOD: геометрия тек train-да; шек бөлек validation-да. Test қолданылмайды.
# Модель training=False: augmentation өшірулі. Бұл X-ray детекторы емес.
try:
    mobile_layers = transfer.layers
    base_index = next(i for i,layer in enumerate(mobile_layers)
                      if isinstance(layer,tf.keras.Model) and layer.name.startswith('mobilenetv2'))
    def ood_features(dataset):
        chunks = []
        for images,_ in dataset:
            x = images
            for layer in mobile_layers[1:base_index+1]:
                x = layer(x,training=False)
            chunks.append(tf.reduce_mean(x,axis=(1,2)).numpy())
        values = np.concatenate(chunks).astype(np.float64)
        if not np.isfinite(values).all():
            raise ValueError('OOD белгілерінде жарамсыз мән бар.')
        return values
    train_vectors = ood_features(make_dataset(train_df))
    validation_vectors = ood_features(val_ds)
    mean = train_vectors.mean(axis=0)
    covariance = np.cov(train_vectors,rowvar=False)
    scale = max(float(np.trace(covariance)/len(mean)),1e-8)
    covariance = .9*covariance + .1*scale*np.eye(len(mean))
    precision = np.linalg.solve(covariance,np.eye(len(mean)))
    precision = (precision+precision.T)/2
    centered = validation_vectors-mean
    distances = np.sqrt(np.maximum(0,np.einsum('ij,jk,ik->i',centered,precision,centered)))
    rank = int(np.ceil((len(distances)+1)*.95))
    if rank>len(distances):
        raise ValueError('Validation жиыны OOD шегін баптауға тым аз.')
    ood_threshold = float(np.sort(distances)[rank-1])
    if not np.isfinite(ood_threshold) or ood_threshold<=0:
        raise ValueError('OOD шегі жарамсыз.')
    # model/config SHA are added only after both files have been serialized.
    ood_parameters = dict(mean=mean,precision=precision,threshold=ood_threshold)
    ood_metadata = dict(schema=2,method='train_validation_mahalanobis_v2',
        reference_split='train',reference_count=len(train_vectors),calibration_split='validation',
        calibration_count=len(validation_vectors),target_acceptance=.95,quantile_rank=rank,
        covariance_shrinkage=.1,test_or_external_used_to_fit=False,clinical_or_formal_coverage_guarantee=False)
    print('OOD: train үлгісі, бөлек validation шегі дайын. Клиникалық/қамту кепілдігі жоқ.')
except Exception as exc:
    ood_parameters = None
    print('OOD қолжетімсіз:',exc)
