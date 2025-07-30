import mmwave as mm
from mmwave.dataloader import DCA1000

# Read data Radar
dca = DCA1000()
adc_data = dca.read()
radar_cube = mm.dsp.range_processing(adc_data)
print("ekel")