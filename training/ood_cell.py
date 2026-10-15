# OOD параметрлері тек validation жиынынан есептеледі.
# Бұл эвристиканың non-X-ray деректеріндегі сапасы бөлек тексерілуі керек.
try:
    mobile_layers = transfer.layers
    base_index = next(i for i,layer in enumerate(mobile_layers)
                      if isinstance(layer,tf.keras.Model) and layer.name.startswith('mobilenetv2'))
    features = []
    for images,_ in val_ds:
        x = images
        for layer in mobile_layers[1:base_index+1]:
            x = layer(x,training=False)
        features.append(tf.reduce_mean(x,axis=(1,2)).numpy())
    vectors = np.concatenate(features).astype(np.float64)
    mean = vectors.mean(axis=0)
    covariance = np.cov(vectors,rowvar=False)
    # Shrinkage fixes singular covariance when n < feature dimension.
    scale = max(float(np.trace(covariance)/len(mean)),1e-8)
    covariance = .9*covariance + .1*scale*np.eye(len(mean))
    precision = np.linalg.solve(covariance,np.eye(len(mean)))
    centered = vectors-mean
    distances = np.sqrt(np.maximum(0,np.einsum('ij,jk,ik->i',centered,precision,centered)))
    ood_threshold = float(np.quantile(distances,.95))
    # Save after the selected model has been serialized, to bind the exact file ID.
    ood_parameters = dict(mean=mean,precision=precision,threshold=ood_threshold)
    print('OOD validation параметрлері есептелді. Non-X-ray валидациясы жүргізілген жоқ.')
except Exception as exc:
    ood_parameters = None
    print('OOD қолжетімсіз:',exc)
