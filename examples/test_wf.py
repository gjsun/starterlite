import numpy as np

import starterlite

wf = starterlite.analysis.WindowFunction()
wf.survey_goemetry = np.array([180, 1, 14])

print('CHECK wf.survey_goemetry:', wf.survey_goemetry)

print('CHECK wf.n_beam:', wf.n_beam)

print('CHECK wf.n_channel:', wf.n_channel)

print('CHECK beam size:', wf.beam_size_at_z(z=5.9, physical=True) * 2.355)

print('CHECK bandwidth_HF_Mpch:', wf.bandwidth_HF_Mpch)

wf.RunAnalyticalWF(L_x=wf.beam_size_at_z(z=5.9, physical=True)*2.355*wf.n_beam, 
				   L_y=wf.beam_size_at_z(z=5.9, physical=True)*2.355, 
				   L_z=wf.bandwidth_HF_Mpch)