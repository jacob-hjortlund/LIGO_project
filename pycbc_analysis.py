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
from scipy.signal import butter, filtfilt, boxcar

# Default values
f_hp = 15
f1 = 43			# Lower frequency for matching
f2 = 300		# Higher frequency for matching
m1 = 41.743     # mass 1, LOSC value
m2 = 29.237     # mass 2, LOSC value
s1 = 0.355      # spin 1, LOSC value
s2 = -0.769     # spin 2, LOSC value

def read_strain(directory, suffix, win_edge, hp_freq=f_hp, crop=0, print_info=False):

	""" Load .gwf file, highpass strain data and remove edges using a Tukey window """

	data = {'H':{}, 'L':{}}
	for ifo in ['H', 'L']:
	    # Read the detector data
	    fname = directory + '%s-%s1' % (ifo, ifo) + suffix + '.gwf'
	    channel_name = '%s1:LOSC-STRAIN'  % ifo
	    data[ifo]['S'] = read_frame(fname, channel_name)
	    data[ifo]['S'] = data[ifo]['S'].crop(crop, crop)

	    # Initial high pass to remove the strong low-frequency signal
	    data[ifo]['S'] = highpass(data[ifo]['S'], hp_freq)

	    # Tukey window to remove time corrupted by the high pass filter
	    sample_rate = data[ifo]['S'].sample_rate
	    duration = int(data[ifo]['S'].duration)
	    win = tukey(duration*sample_rate, alpha= float(win_edge)/float(duration))
	    data[ifo]['S']._data = data[ifo]['S']._data*win
	    
	    # Print some information
	    if print_info:
		    print(' ')
		    print('****************************************************')
		    print("%s-file is: %s" %(ifo, fname))
		    print("%s-file is from %s to %s, duration: %s" 
		        %(ifo, data[ifo]['S'].start_time, data[ifo]['S'].end_time, duration) )

	    # Also create a frequency domain version of the data
	    data[ifo]['ST'] = data[ifo]['S'].to_frequencyseries()

	return data

def estimate_psd(data, psd_segment_length, low_freq=f1, psd_method='median', psd_window='hann', kaiser_beta=14):

	""" Calculate the PSD using a Welch-style estimator, and then interpolate the
	PSD to the desired frequency step. """

	data_tmp = {'H':{}, 'L':{}}
	for ifo in ['H', 'L']:

		data_tmp[ifo]['S'] = data[ifo]['S']
		data_tmp[ifo]['ST'] = data[ifo]['ST']

		sample_rate = data_tmp[ifo]['S'].sample_rate
		if psd_window == 'blackman':
			blackman_window = np.blackman(psd_segment_length*sample_rate)
			data_tmp[ifo]['PSD'] = interpolate(data_tmp[ifo]['S'].psd(psd_segment_length, 
			avg_method=psd_method, window=blackman_window), data_tmp[ifo]['ST'].delta_f)
		elif psd_window == 'hann':
			data_tmp[ifo]['PSD'] = interpolate(data_tmp[ifo]['S'].psd(psd_segment_length, 
			avg_method=psd_method, window=psd_window), data_tmp[ifo]['ST'].delta_f)
		elif psd_window == 'kaiser':
			kaiser_window = np.kaiser(psd_segment_length*sample_rate, kaiser_beta)
			data_tmp[ifo]['PSD'] = interpolate(data_tmp[ifo]['S'].psd(psd_segment_length, 
			avg_method=psd_method, window=kaiser_window), data_tmp[ifo]['ST'].delta_f)
		elif psd_window == 'rectangular':
			rectangular_window = boxcar(psd_segment_length*sample_rate)
			data_tmp[ifo]['PSD'] = interpolate(data_tmp[ifo]['S'].psd(psd_segment_length, 
			avg_method=psd_method, window=rectangular_window), data_tmp[ifo]['ST'].delta_f)

		# Smooth to the desired corruption length

		data_tmp[ifo]['PSD'] = inverse_spectrum_truncation(data_tmp[ifo]['PSD'], 
			psd_segment_length*sample_rate,low_frequency_cutoff=low_freq, trunc_method='hann')

		# Whiten strain

		data_tmp[ifo]['WS'] = (data_tmp[ifo]['ST']/data_tmp[ifo]['PSD']**0.5).to_timeseries()

	return data_tmp

def matched_filtering(data, m1=m1,m2=m2,s1=s1,s2=s2,f1=f1,f2=f2,
					  bp=True, bpf1=f1, bpf2=f2, print_info=False):
	
	""" bla """

	duration = data['H']['S'].duration
	sample_rate = data['H']['S'].sample_rate
	delta_t = data['H']['S'].delta_t
	delta_f = data['H']['ST'].delta_f
	nq = float(sample_rate)/2.
	b, a = butter(4, [float(bpf1)/nq, float(bpf2)/nq], btype = 'bandpass')

	# Find td waveform for increased precision
	hp, hc = get_td_waveform(approximant="SEOBNRv4", 
                         mass1=m1, mass2=m2, 
                         spin1z = s1,      spin2z = s2, 
                         f_lower=f1, delta_t = data['H']['S'].delta_t)
	hp1 = hp.copy()
	# Convert to freq. series
	len0 = len(hp)
	hp.append_zeros(int(len0*0.1))
	hp._data = hp._data*tukey(len(hp), alpha=0.1 )
	hp = hp.to_frequencyseries(delta_f=delta_f)
	hp = apply_fseries_time_shift(hp, duration/2.-len0*delta_t/1.1)
	hp.resize( int( np.floor( int(duration*sample_rate)/2) + 1 ) )

	# Start matched filtering

	max_snr, max_time, max_phase = {}, {}, {}
	for ifo in ['H', 'L']:

		tpl = hp.copy()

		snr = matched_filter(hp, data[ifo]['ST'], psd=data[ifo]['PSD'],
		 low_frequency_cutoff=f1)

		_, idx = snr.abs_max_loc()
		max_time[ifo] = float(idx) / snr.sample_rate + snr.start_time
		max_snr[ifo] = snr[idx]
		max_phase[ifo] = np.angle(max_snr[ifo])
		fac = max_snr[ifo] / sigma(tpl, psd=data[ifo]['PSD'], low_frequency_cutoff=f1, high_frequency_cutoff=f2)
		dt =  max_time[ifo] - data[ifo]['ST'].start_time

		if print_info:
		    print(' ')
		    print('****************************************************')
		    print('%s: Consider SNR only from %s to %s, duration: %s' 
		    	%(ifo, snr.start_time, snr.end_time, snr.duration))
		    print('%s-SNR: %s'   %(ifo, np.absolute(max_snr[ifo])))
		    print('%s-time: %s'  %(ifo, max_time[ifo]))
		    print('%s-phase: %s' %(ifo, max_phase[ifo]))

		# Shift the template to the maximum time at this sample rate
		tpl = apply_fseries_time_shift(tpl, dt) * fac

		# Find freq. domain residual
		rtilde = data[ifo]['ST']-tpl
		data[ifo]['TPLS'] = tpl

		data[ifo]['R'] = rtilde.to_timeseries()
		data[ifo]['TPL'] = tpl.to_timeseries()

		# Whiten residual and template
		residual = (rtilde / data[ifo]['PSD'] ** 0.5).to_timeseries()
		tpl = (tpl / data[ifo]['PSD'] ** 0.5).to_timeseries()

		data[ifo]['WR'] = residual
		data[ifo]['WTPL'] = tpl

		if bp:
			if print_info:
				print(' ')
				print('****************************************************')
				print('Bandpassing %s-data from %s Hz to %s Hz.' %(ifo, bpf1, bpf2))
			strain_tmp = data[ifo]['S'].copy()
			strain_tmp._data = filtfilt(b,a,data[ifo]['WS']._data) 
			residual._data = filtfilt(b,a,residual._data)
			tpl._data = filtfilt(b,a,tpl._data)
			data[ifo]['WBPS'] = strain_tmp
			data[ifo]['WBPR'] = residual
			data[ifo]['WBPTPL'] = tpl


	network_snr = (abs(np.array(max_snr.values())) ** 2.0).sum() ** 0.5

	if print_info:
		print('Network SNR: %s' %(network_snr))

	return data

