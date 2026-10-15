# Validation арқылы temperature scaling. Test бұл есептеуге қатыспайды.
from scipy.optimize import minimize_scalar
from scipy.special import expit

def temperature_probabilities(probabilities, temperature):
    probabilities = np.clip(np.asarray(probabilities,dtype=np.float64),1e-7,1-1e-7)
    logits = np.log(probabilities)-np.log1p(-probabilities)
    return expit(logits/temperature)

calibrations = {}
calibration_diagnostics = []
for name,model in models.items():
    raw_val = model.predict(val_ds,verbose=0).ravel()
    val_y = val_df.label.to_numpy()
    logits = np.log(np.clip(raw_val.astype(np.float64),1e-7,1-1e-7))-np.log1p(-np.clip(raw_val.astype(np.float64),1e-7,1-1e-7))
    def nll(temperature):
        z = logits/temperature
        return float(np.mean(np.logaddexp(0,z)-val_y*z))
    fit = minimize_scalar(nll,bounds=(.2,5.),method='bounded')
    if not fit.success:
        raise ValueError('Калибрлеу оңтайландыруы аяқталмады.')
    temperature = float(fit.x)
    calibrated_val = temperature_probabilities(raw_val,temperature)
    threshold,table = choose_threshold(val_y,calibrated_val)
    thresholds[name] = threshold
    calibrations[name] = {'method':'temperature','T':temperature}
    table.to_csv(OUT/f'{name}_validation_calibrated_thresholds.csv',index=False)
    calibration_diagnostics.append({'model':name,'T':temperature,'validation_NLL_raw':nll(1.),'validation_NLL_calibrated':nll(temperature),'threshold':threshold})
pd.DataFrame(calibration_diagnostics).to_csv(OUT/'calibration_validation.csv',index=False)
display(pd.DataFrame(calibration_diagnostics))
