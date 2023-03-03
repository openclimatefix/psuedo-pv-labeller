from psuedo_pv_labeller.model import PsuedoIrradienceForecastor
import torch
import torch.nn.functional as F

def test_irradience_forecastor_init():
    model = PsuedoIrradienceForecastor()

def test_irradience_forecastor_forward():
    model = PsuedoIrradienceForecastor()
    input_tensor = torch.rand((2, 3, 12, 128, 128))
    output_tensor = model(input_tensor)
    assert not torch.any(torch.isnan(output_tensor))

def test_irradience_forecastor_backward():
    model = PsuedoIrradienceForecastor()
    input_tensor = torch.rand((2, 3, 12, 128, 128))
    target_tensor = torch.rand((2, 1, 128, 128))
    output_tensor = model(input_tensor)
    loss = F.mse_loss(output_tensor, target_tensor)
    assert not torch.any(torch.isnan(loss))
    loss.backward()
