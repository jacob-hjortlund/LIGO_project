import lal as _lal
import numpy as np
from pycbc.frame import read_frame
from pycbc.filter import highpass
from scipy.signal import tukey
from pycbc.filter import sigma
from pycbc.types.timeseries import TimeSeries
from pycbc.types.timeseries import load_timeseries
from pycbc.psd import interpolate, inverse_spectrum_truncation
from pycbc.waveform import get_td_waveform
from pycbc.filter import matched_filter
from pycbc.waveform import apply_fseries_time_shift
from scipy.signal import butter, filtfilt, boxcar

from pycbc.waveform.generator import FDomainCBCGenerator, FDomainDetFrameGenerator, TDomainCBCGenerator
from pycbc.waveform import get_td_waveform, get_fd_waveform, apply_fseries_time_shift

# Default values
f_hp = 15
tpl_f = 20
f1 = 43			# Lower frequency for matching
f2 = 300		# Higher frequency for matching
m1 = 41.743     # mass 1, LOSC value
m2 = 29.237     # mass 2, LOSC value
s1 = 0.355      # spin 1, LOSC value
s2 = -0.769     # spin 2, LOSC value

def read_strain(directory, suffix, win_edge, sample_rate=4096, hp_freq=f_hp, crop_left=0, crop_right=0, print_info=False, losc=True, gwf=True):

	""" Load .gwf file, highpass strain data and remove edges using a Tukey window """

	data = {'H':{}, 'L':{}}
	for ifo in ['H', 'L']:
	    # Read the detector data
	    if gwf:

	    	fname = directory + '%s-%s1' % (ifo, ifo) + suffix + '.gwf'
	    	if losc:
	    		channel_name = '%s1:LOSC-STRAIN'  % ifo
	    	else:
	    		channel_name = '%s1:GWOSC-%sKHZ_R1_STRAIN' %(ifo, sample_rate/1024)
	    	data[ifo]['S'] = read_frame(fname, channel_name)
	    
	    else:

	    	fname = directory + '%s_%sKHZ/%s_%s' %(ifo, sample_rate/1024, ifo, suffix)
	    	tpl_inj = np.load('Templates/%s_%sKHZ.npy' %(ifo, sample_rate/1024))
	    	data_tmp = load_timeseries(fname)
	    	data[ifo]['SR'] = data_tmp.copy()
	    	data_tmp._data += tpl_inj
	    	data[ifo]['S'] = data_tmp

	    
	    data[ifo]['S'] = data[ifo]['S'].crop(crop_left, crop_right)
	    data[ifo]['SR'] = data[ifo]['SR'].crop(crop_left, crop_right)

	    # Initial high pass to remove the strong low-frequency signal
	    if hp_freq != 0:
	    	data[ifo]['S'] = highpass(data[ifo]['S'], hp_freq)
	    	data[ifo]['SR'] = highpass(data[ifo]['SR'], hp_freq)

	    # Tukey window to remove time corrupted by the high pass filter
	    sample_rate = data[ifo]['S'].sample_rate
	    duration = int(data[ifo]['S'].duration)
	    win = tukey(duration*sample_rate, alpha= float(win_edge)*2./float(duration))
	    data[ifo]['S']._data = data[ifo]['S']._data*win
	    data[ifo]['SR']._data = data[ifo]['SR']._data*win
	    
	    # Print some information
	    if print_info:
		    print(' ')
		    print('****************************************************')
		    print("%s-file is: %s" %(ifo, fname))
		    print("%s-file is from %s to %s, duration: %s" 
		        %(ifo, data[ifo]['S'].start_time, data[ifo]['S'].end_time, duration) )

	    # Also create a frequency domain version of the data
	    data[ifo]['ST'] = data[ifo]['S'].to_frequencyseries()
	    data[ifo]['SRT'] = data[ifo]['SR'].to_frequencyseries()

	return data

def estimate_psd(data, psd_segment_length, low_freq=f1, psd_method='median', psd_window='hann', kaiser_beta=14):

	""" Calculate the PSD using a Welch-style estimator, and then interpolate the
	PSD to the desired frequency step. """

	data_tmp = {'H':{}, 'L':{}}
	for ifo in ['H', 'L']:

		data_tmp[ifo]['S'] = data[ifo]['S']
		data_tmp[ifo]['SR'] = data[ifo]['SR']
		data_tmp[ifo]['ST'] = data[ifo]['ST']
		data_tmp[ifo]['SRT'] = data[ifo]['SRT']

		sample_rate = data_tmp[ifo]['S'].sample_rate
		if psd_window == 'blackman':
			blackman_window = np.blackman(psd_segment_length*sample_rate)
			
			data_tmp[ifo]['PSD'] = interpolate(data_tmp[ifo]['S'].psd(psd_segment_length, 
			avg_method=psd_method, window=blackman_window), data_tmp[ifo]['ST'].delta_f)

			data_tmp[ifo]['SR_PSD'] = interpolate(data_tmp[ifo]['SR'].psd(psd_segment_length, 
			avg_method=psd_method, window=blackman_window), data_tmp[ifo]['SRT'].delta_f)

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

		data_tmp[ifo]['SR_PSD'] = inverse_spectrum_truncation(data_tmp[ifo]['SR_PSD'], 
			psd_segment_length*sample_rate,low_frequency_cutoff=low_freq, trunc_method='hann')

		# Whiten strain

		data_tmp[ifo]['WS'] = (data_tmp[ifo]['ST']/data_tmp[ifo]['PSD']**0.5).to_timeseries()
		data_tmp[ifo]['WSR'] = (data_tmp[ifo]['SRT']/data_tmp[ifo]['PSD']**0.5).to_timeseries()

	return data_tmp

def make_tpls_fd(tpl_type, sample_rate, duration, f1=10.0, f_ref=20.0, m1=m1, m2=m2, s1=s1, s2=s2):
    """Make GW template with given sampling rate and max-duration. 
    The default is to use the LOSC parameters for GW150914.
    ----------------------------------
    Parameters
    -----------------------------------
    tpl_type: 1 = LOSC parameters+IMRPhenomPv2, 2 = LOSC parameters+SEOBNRv4,
    3 = Alex's ML-template

    sample_rate: the sample rate

    duration: expected duration of the template in seconds. Should be bigger
    than the intrinsic template length

    The default values for m1, m2, s1, s2 are the LOSC Parameters for GW150914

    f1: the lower frequency cut-off of the GW template

    f_ref: used by Alex's ML-template. Use the default value is fine. 
    """    
    delta_t = 1./sample_rate
    delta_f = 1./duration
    len_td = int(duration*sample_rate)
    len_fd = int(np.floor(len_td/2) + 1)
    
    # LOSC template, IMRPhenomPv2
    if tpl_type==1:
        hp, hc = get_fd_waveform(approximant="IMRPhenomPv2",mass1=m1,mass2=m2,spin1z=s1,spin2z=s2,f_lower=f1,delta_f=delta_f)
        hp.resize(len_fd)
        hp = apply_fseries_time_shift(hp, -duration/2) # put the peak roughly at middle
        hp = {'H':hp,'L':hp}

    # LOSC template, SEOBNRv4
    if tpl_type==2:
        hp, hc = get_td_waveform(approximant='SEOBNRv4',mass1=m1,mass2=m2,spin1z=s1,spin2z=s2,delta_t=delta_t,f_lower=f1)
        len0 = len(hp)
        hp.append_zeros( int( len(hp)*0.1 ) )   # append some zeros so the chirp is unaffected
        hp._data = hp._data*tukey(len(hp),alpha=0.1)
        hp = hp.to_frequencyseries(delta_f=delta_f)
        hp = apply_fseries_time_shift(hp, duration/2 - len0*delta_t/1.1) # put the peak roughly at middle
        hp.resize(len_fd)
        hp = {'H':hp,'L':hp}

    # The Alex ML-template
    if tpl_type==3:
        static_args =  {    'approximant': 'IMRPhenomPv2',
                            'f_lower': f1,
                            'f_ref': f_ref}
        variable_args= [    'tc', 'mass1', 'mass2', 
                            'spin1_a', 'spin1_azimuthal', 'spin1_polar', 
                            'spin2_a', 'spin2_azimuthal', 'spin2_polar', 
                            'distance', 'coa_phase', 'inclination', 'polarization', 'ra', 'dec']
        maxL_params = {
            'coa_phase': 0.68832125393756094,
            'dec': -1.2734810548587012,
            'distance': 476.75645472243809,
            'inclination': 2.913271378217019,
            'mass1': 39.025765698169494,
            'mass2': 32.062563121887692,
            'polarization': 5.9925231806647368,
            'ra': 1.5730257459180903,
            'spin1_a': 0.97679616619854848,
            'spin1_azimuthal': 3.6036952308172112,
            'spin1_polar': 1.6283548507561103,
            'spin2_a': 0.18876088159164967,
            'spin2_azimuthal': 3.4359460623268951,
            'spin2_polar': 2.4915268979667631,
            'tc': 1126259462.41756463050842285,
        }
        generator = FDomainDetFrameGenerator(FDomainCBCGenerator, 1126259462-4, 
            detectors=['H1', 'L1'], variable_args=variable_args, 
            delta_f=delta_f, delta_t=delta_t, **static_args)
        htildes = generator.generate(**{p: maxL_params[p] for p in variable_args})
        for ifo, h in htildes.items():
            htildes[ifo] = h * np.exp(1j * (-0.915527673))
        tplH = htildes['H1']
        tplH.resize(len_fd)
        tplL = htildes['L1']
        tplL.resize(len_fd)
        hp = {'H':tplH, 'L':tplL}

    return hp


def matched_filtering(data, m1=m1,m2=m2,s1=s1,s2=s2,f1=f1,f2=f2, tpl_f=tpl_f,
					  bp=True, bpf1=f1, bpf2=f2, print_info=False, approximant='SEOBNRv4', inj=False, tpl_type=3):
	
	""" bla """

	duration = data['H']['S'].duration
	sample_rate = data['H']['S'].sample_rate
	delta_t = data['H']['S'].delta_t
	delta_f = data['H']['ST'].delta_f
	nq = float(sample_rate)/2.
	b, a = butter(4, [float(bpf1)/nq, float(bpf2)/nq], btype = 'bandpass')

	# Find td waveform for increased precision

	if not inj:

		hp, hc = get_td_waveform(approximant=approximant, 
                     mass1=m1, mass2=m2, 
                     spin1z = s1,      spin2z = s2, 
                     f_lower=tpl_f, delta_t = data['H']['S'].delta_t)
		hp1 = hp.copy()
		# Convert to freq. series
		len0 = len(hp)
		hp.append_zeros(int(len0*0.1))
		hp._data = hp._data*tukey(len(hp), alpha=0.1 )
		hp = hp.to_frequencyseries(delta_f=delta_f)
		hp = apply_fseries_time_shift(hp, duration/2.-len0*delta_t/1.1)
		hp.resize( int( np.floor( int(duration*sample_rate)/2) + 1 ) )

	else:

		fp = make_tpls_fd(tpl_type, sample_rate, 32, m1=m1, m2=m2, s1=s1, s2=s2)
		print(type(fp['H']))
		print(type(fp['L']))

	# Start matched filtering

	max_snr, max_time, max_phase = {}, {}, {}
	for ifo in ['H', 'L']:

		if not inj:

			tpl = hp.copy()

		else:

			hp = fp[ifo]
			tpl = fp[ifo].copy()
			print(type(tpl))

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
			strain_tmp = data[ifo]['WS'].copy()
			raw_strain_tmp = data[ifo]['WSR'].copy()
			strain_tmp._data = filtfilt(b,a,data[ifo]['WS']._data)
			raw_strain_tmp._data = filtfilt(b,a,data[ifo]['WSR']._data)
			residual._data = filtfilt(b,a,residual._data)
			tpl._data = filtfilt(b,a,tpl._data)
			data[ifo]['WBPS'] = strain_tmp
			data[ifo]['WBPSR'] = raw_strain_tmp
			data[ifo]['WBPR'] = residual
			data[ifo]['WBPTPL'] = tpl


	network_snr = (abs(np.array(max_snr.values())) ** 2.0).sum() ** 0.5

	if print_info:
		print('Network SNR: %s' %(network_snr))

	return data

