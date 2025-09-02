import torch
import torch_xla

a_golden = torch.arange(12, device="cpu").reshape(3, 4)
b_golden, c_golden = a_golden.split([3, 1], dim=-1)
a_xla = torch.arange(12, device="xla").reshape(3, 4)
b_xla, c_xla = a_xla.split([3, 1], dim=-1)

print("a original:", a_golden)
print("b golden :", b_golden)
print("b xla :", b_xla)
print("c golden :", c_golden)
print("c xla :", c_xla)

torch.testing.assert_close(b_golden, b_xla.cpu(), rtol=0, atol=0)
torch.testing.assert_close(c_golden, c_xla.cpu(), rtol=0, atol=0)