import lal as _lal
import numpy as np
from pycbc.frame import read_frame
from pycbc.filter import highpass
from scipy.signal import tukey
from pycbc.filter import sigma
from pycbc.types.timeseries import TimeSeries
from pycbc.psd import interpolate, inverse_spectrum_truncation
from pycbc.waveform import get_td_waveform
from pycbc.filter import matched_filter
from pycbc.waveform import apply_fseries_time_shift
from scipy.signal import butter, filtfilt

# Default values
f_tpl = 10.0    # lower frequency for templates
f_hp = 15
f1 = 43			# Lower frequency for matching
f2 = 300		# Higher frequency for matching
m1 = 41.743     # mass 1, LOSC value
m2 = 29.237     # mass 2, LOSC value
s1 = 0.355      # spin 1, LOSC value
s2 = -0.769     # spin 2, LOSC value

def read_strain(directory, suffix, win_edge, hp_freq=f_hp, print_info=False):

	""" Load .gwf file, highpass strain data and remove edges using a Tukey window """

	strain, stilde = {}, {}
	for ifo in ['H', 'L']:
	    # Read the detector data
	    fname = directory + '%s-%s1' % (ifo, ifo) + suffix + '.gwf'
	    channel_name = '%s1:LOSC-STRAIN'  % ifo
	    strain[ifo] = read_frame(fname, channel_name)

	    # Initial high pass to remove the strong low-frequency signal
	    strain[ifo] = highpass(strain[ifo], hp_freq)

	    # Tukey window to remove time corrupted by the high pass filter
	    sample_rate = strain[ifo].sample_rate
	    duration = int(strain[ifo].duration)
	    win = tukey(duration*sample_rate, alpha= float(win_edge)/float(duration))
	    strain[ifo]._data = strain[ifo]._data*win
	    
	    # Print some information
	    if print_info:
		    print(' ')
		    print('****************************************************')
		    print("%s-file is: %s" %(ifo, fname))
		    print("%s-file is from %s to %s, duration: %s" 
		        %(ifo, strain[ifo].start_time, strain[ifo].end_time, duration) )

	    # Also create a frequency domain version of the data
	    stilde[ifo] = strain[ifo].to_frequencyseries()

	return {'H': {'S': strain['H'], 'ST': stilde['H']},
		   'L': {'S': strain['L'], 'ST': stilde['L']}}

def estimate_psd(data, psd_segment_length, low_freq=f1, psd_method='median', psd_window='hann'):

	""" Calculate the PSD using a Welch-style estimator, and then interpolate the
	PSD to the desired frequency step. """

	for ifo in ['H', 'L']:

		sample_rate = data[ifo]['S'].sample_rate
		#psd_window = np.blackman(psd_segment_length*sample_rate)
		psd = interpolate(data[ifo]['S'].psd(psd_segment_length, avg_method=psd_method,
		 window=psd_window), data[ifo]['ST'].delta_f)

		# Smooth to the desired corruption length

		psd = inverse_spectrum_truncation(psd, psd_segment_length*sample_rate,
			low_frequency_cutoff=low_freq, trunc_method='hann')

		# Whiten strain

		data[ifo]['S'] = (data[ifo]['ST']/psd**0.5).to_timeseries()

		data[ifo]['PSD'] = psd

	return data

def matched_filtering(data, m1=m1,m2=m2,s1=s1,s2=s2,f_tpl=f_tpl,f1=f1,f2=f2,
					  bp=True, bpf1=f1, bpf2=f2, print_info=False):
	
	""" bla """

	duration = data['H']['S'].duration
	sample_rate = data['H']['S'].sample_rate
	delta_t = data['H']['S'].delta_t
	delta_f = data['H']['S'].delta_f
	nq = float(sample_rate)/2.
	b, a = butter(4, [float(bpf1)/nq, float(bpf2)/nq], btype = 'bandpass')

	# Find td waveform for increased precision
	hp, hc = get_td_waveform(approximant="SEOBNRv4", 
                         mass1=m1, mass2=m2, 
                         spin1z = s1,      spin2z = s2, 
                         f_lower=f_tpl, delta_t = data['H']['S'].delta_t)

	# Convert to freq. series
	len0 = len(hp)
	hp.append_zeros(int(len0*0.1))
	hp._data = hp._data*tukey(len(hp), alpha=0.1 )
	hp = hp.to_frequencyseries(delta_f=delta_f)
	hp = apply_fseries_time_shift(hp, duration/2.-len0*delta_t/1.1)
	hp.resize( int( np.floor(int(duration*sample_rate)/2) + 1 ) )

	# Start matched filtering

	max_snr, max_time, max_phase = {}, {}, {}
	for ifo in ['H', 'L']:

		snr = matched_filter(hp, data[ifo]['ST'], psd=data[ifo]['PSD'],
		 low_frequency_cutoff=f1, high_frequency_cutoff=f2)

		_, idx = snr.abs_max_loc()
		max_snr[ifo] = snr[idx]
		max_time[ifo] = float(idx) /snr.sample_rate + snr.start_time
		max_phase[ifo] = np.angle(max_snr[ifo])

		# Shift the template to the maximum time at this sample rate
		dt = max_time[ifo]-data[ifo]['ST'].start_time
		tpl = apply_fseries_time_shift(hp, dt)

		# Scale the template to SNR and phase
		tpl /= sigma(hp, psd=data[ifo]['PSD'], low_frequency_cutoff=f1)
		tpl *= max_snr[ifo]

		amp = max_snr[ifo] / sigma(hp, psd=data[ifo]['PSD'], low_frequency_cutoff=f1)
		amp = np.absolute(amp)

		# Find freq. domain residual
		rtilde = data[ifo]['ST']-tpl

		# Whiten residual and template
		residual = (rtilde / data[ifo]['ST'] ** 0.5).to_timeseries()
		tpl = (tpl / data[ifo]['ST'] **0.5).to_timeseries()

		if bp:
			if print_info:
				print(' ')
				print('****************************************************')
				print('Bandpassing %s-data from %s Hz to %s Hz.' %(ifo, bpf1, bpf2))
			data[ifo]['S']._data = filtfilt(b,a,data[ifo]['S']._data) 
			residual._data = filtfilt(b,a,residual._data)
			tpl._data = filtfilt(b,a,tpl._data)

		data[ifo]['R'] = residual
		data[ifo]['TPL'] = tpl

		if print_info:
		    print(' ')
		    print('****************************************************')
		    print('%s: Consider SNR only from %s to %s, duration: %s' 
		    	%(ifo, snr.start_time, snr.end_time, snr.duration))
		    print('%s-SNR: %s'   %(ifo, np.absolute(max_snr[ifo])))
		    print('%s-time: %s'  %(ifo, max_time[ifo]))
		    print('%s-phase: %s' %(ifo, max_phase[ifo]))
		    print('%s relative amplitude: %s' %(ifo, amp*100))


	network_snr = (abs(np.array(max_snr.values())) ** 2.0).sum() ** 0.5

	if print_info:
		print('Network SNR: %s' %(network_snr))

	return data

