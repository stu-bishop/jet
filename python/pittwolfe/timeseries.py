"""
    timeseries
    -----------
    A collection of routines for time series analysis
"""

import warnings
import re
import numpy as np
from numpy import pi
from numba import jit, njit, prange
import scipy as sp
import scipy.signal
import pandas as pd
import xarray as xr
from .util import flatten

##########################################################################################
# Calendar functions
##########################################################################################

dpm = {'noleap': [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
       '365_day': [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
       'standard': [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
       'gregorian': [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
       'proleptic_gregorian': [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
       'all_leap': [0, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
       '366_day': [0, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
       '360_day': [0, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30]}

def leap_year(year, calendar='standard'):
    '''
    Determine if year is a leap year

    Parameters
    ----------
    year : integer
        Year in question.
    calendar : str, optional
        Calendar to use. Case is ignored. Default: standard.

    Returns
    -------
    leap : bool
        True if the year is a leap year, false otherwise.

    Notes
    -----
    Supported calendars are 'standard', 'Gregorian', 'Proleptic_Gregorian', and 'Julian'.
    'Standard' is the same as 'Gregorian'.

    '''

    leap = False
    if ((calendar.lower() in ['standard', 'gregorian',
        'proleptic_gregorian', 'julian']) and
        (year % 4 == 0)):
        leap = True
        if ((calendar.lower() == 'proleptic_gregorian') and
            (year % 100 == 0) and
            (year % 400 != 0)):
            leap = False
        elif ((calendar.lower() in ['standard', 'gregorian']) and
                 (year % 100 == 0) and (year % 400 != 0) and
                 (year < 1583)):
            leap = False
    return leap



def get_dpm(time, calendar='standard'):
    '''
    return a array of days per month corresponding to the months provided in `time`.

    Parameters
    ----------
    time : DatatimeIndex, MultiIndex, tuple, list, ndarray
        Dates in question.
        If MultiIndex, the first level should be the year and the second the month.
        if tuple or list, should be in the format (year, month).
        If ndarray, first column should be years, second column should be months.
    calendar : str, optional
        Calendar to use. Case is ignored. Default: standard.

    Returns
    -------
    month_length : array of int
        Days in each month listed in `time`.

    Notes
    -----
    Supported calendars are 'standard', 'Gregorian', 'Proleptic_Gregorian', and 'Julian'.
    'Standard' is the same as 'Gregorian'.

    '''
    cal_days = dpm[calendar]

    if isinstance(time, pd.DatetimeIndex):
        years  = time.year
        months = time.month
    elif isinstance(time, pd.MultiIndex):
        years  = time.get_level_values(0)
        months = time.get_level_values(1)
    elif isinstance(time, tuple) or isinstance(time, list):
        years  = (time[0], )
        months = (time[1], )
    elif isinstance(time, np.ndarray):
        years  = time[:,0]
        months = time[:,1]
    elif isinstance(time, pd.core.frame.DataFrame):
        years  = time.filter(regex=re.compile('^year$', re.I)).values[:,0]
        months = time.filter(regex=re.compile('^month$', re.I)).values[:,0]
    else:
        raise RuntimeError('Don\'t know how to parse the \'time\' input.')

    month_length = np.zeros(len(years), dtype=np.int)

    for i, (month, year) in enumerate(zip(months, years)):
        month_length[i] = cal_days[month]
        if month == 2 and leap_year(year, calendar=calendar):
            month_length[i] += 1

    if isinstance(time, pd.core.frame.DataFrame):
        return pd.Series(month_length, index=time.index)
    else:
        return month_length

##########################################################################################
# Means and standardization
##########################################################################################

def standardize(x, axis=0):
    y = x - np.nanmean(x, axis=axis, keepdims=True)
#     y /= np.nanstd(y, axis=axis, keepdims=True)

    ystd = np.nanstd(y, axis=axis, keepdims=True)
    recip_ystd = np.zeros_like(ystd)
    ix = ystd > 0
    recip_ystd[ix] = 1/ystd[ix]
    y *= recip_ystd

    return y

# ----------------------------------------------------------------------------------------
def seasonalMean(data_in, theSeasons, t=None):
    '''
    Performs a mean over the seasons given in the "seasons" array, which is a 12 element
    array with an entry for each month. Non-zero months are averaged over.
    Negative entries correspond to months from the previous year. For example, a DJF
    mean would have

       seasons = [ 1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  -1]

    Alternately, common seasons can be specified, such as 'djf', 'mam', etc

    The data should have dimensions NYear x 12 x ... or 12*NYear x ...
    '''

    season_map = {
        'annual': [ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1],
        'ann':    [ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1],
        'djf':    [ 1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1],
        'mam':    [ 0,  0,  1,  1,  1,  0,  0,  0,  0,  0,  0,  0],
        'jja':    [ 0,  0,  0,  0,  0,  1,  1,  1,  0,  0,  0,  0],
        'son':    [ 0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  0],
        'jfm':    [ 1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        'amj':    [ 0,  0,  0,  1,  1,  1,  0,  0,  0,  0,  0,  0],
        'jas':    [ 0,  0,  0,  0,  0,  0,  1,  1,  1,  0,  0,  0],
        'ond':    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1],
        'djfm':   [ 1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0, -1],
        'amjj':   [ 0,  0,  0,  1,  1,  1,  1,  0,  0,  0,  0,  0],
        'ason':   [ 0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  0],
        }

    if isinstance(theSeasons, str):
        theSeasons = theSeasons.casefold()
        try:
            seasons = season_map[theSeasons]
        except KeyError:
            raise KeyError('Currently available seasons are ' + season_map.keys())
    else:
        seasons = theSeasons

#     from IPython.core.debugger import Tracer
#     Tracer()()
    seasons = np.array(seasons)

    ix_neg = np.nonzero(seasons < 0)[0]

    data = data_in.copy()

    if data.ndim < 2 or data.shape[1] != 12:
        # reshape so months are the second dimension
        NYear = data.shape[0]//12
        data = data.reshape(flatten(((NYear, 12), data.shape[1:])))

    if len(ix_neg) == 0:
        if t is not None:
            t_seasonal = t

        ix_pos = np.nonzero(seasons > 0)[0]

        data_seasonal = data[:, ix_pos].mean(axis=1)
    else:
        if t is not None:
            t_seasonal = t[1:]

        data_neg = data[:-1, ix_neg].sum(axis=1)

        ix_pos = np.nonzero(seasons > 0)[0]
        data_pos = data[1:, ix_pos].sum(axis=1)

        data_seasonal = (data_pos + data_neg)/(len(ix_neg) + len(ix_pos))

    if t is None:
        return data_seasonal
    else:
        return t_seasonal, data_seasonal

# ----------------------------------------------------------------------------------------
def seasonal_mean(src, atm=False, mask_null=False):
    '''
    Produce seasonal means

    Parameters
    ----------
    src : XArray DataArray or DataSet or Pandas DataFrame
    atm : bool, options
        If true, use the "atmospheric seasons" DJF, MAM, JJA, SON. Otherwise, use the "oceanic seasons"
            JFM, AMJ, JAS, OND.

    Returns
    -------
    means : XArray DataArray or DataSet or Pandas DataFrame depending on input type.
        Seasonal means.


    '''
    if atm:
        quarter = 'Q-NOV'
    else:
        quarter = 'Q-DEC'

    if isinstance(src, xr.Dataset) or isinstance(src, xr.DataArray):
        month_length = src.time.dt.days_in_month
#         month_length = xr.DataArray(get_dpm(src.time.to_index(), calendar=calendar),
#                                     coords=[src.time], name='month_length')

        if mask_null:
            season_length = month_length.where(~src.isnull()).resample(time=quarter).sum()
        else:
            season_length = month_length.resample(time=quarter).sum()
        fld = (src * month_length).resample(time=quarter, restore_coord_dims=True).sum(dim='time') / season_length
        fld = fld.where(season_length > 89)

        # shift time index to middle of season
        time = pd.to_datetime(fld.time.values) - pd.DateOffset(months=1)
        fld.coords['time'] = time - pd.to_timedelta(time.day-1, unit='d') + pd.DateOffset(days=14)

    elif isinstance(src, pd.DataFrame):
        month_length = pd.Series(get_dpm(src.index), index=src.index)

        season_length = month_length.resample(quarter).sum()
        fld = src.multiply(month_length, axis='index').resample('Q-NOV').sum().divide(season_length, axis='index')
        fld.loc[season_length < 90] = np.nan

        time = fld.index - pd.DateOffset(months=1)
        fld.index = time - pd.to_timedelta(time.day-1, unit='d') + pd.DateOffset(days=14)


    return fld

# ----------------------------------------------------------------------------------------
def oceanic_seasonal_mean(src, calendar='standard'):
    '''
    This requires the input to be in xarray format.
    '''
    month_length = xr.DataArray(get_dpm(src.time.to_index(), calendar=calendar),
                                coords=[src.time], name='month_length')

    quarter = 'Q-DEC'
    fld = ((src * month_length).resample(time=quarter).sum(dim='time')
             / month_length.resample(time=quarter).sum())
    fld.name = src.name

    # shift time index to middle of season
    time = pd.to_datetime(fld.time.values) - pd.DateOffset(months=1)
    fld.coords['time'] = time - pd.to_timedelta(time.day-1, unit='d') + pd.DateOffset(days=14)

    return fld

# ----------------------------------------------------------------------------------------
def atmospheric_seasonal_mean(src, calendar='standard'):
    '''
    This requires the input to be in xarray format.
    '''
    month_length = xr.DataArray(get_dpm(src.time.to_index(), calendar=calendar),
                                coords=[src.time], name='month_length')

    quarter = 'Q-NOV'

    fld = ((src * month_length).resample(time=quarter).sum(dim='time')
                 / month_length.resample(time=quarter).sum())
    fld.name = src.name

    # shift time index to middle of season
    time = pd.to_datetime(fld.time.values) - pd.DateOffset(months=1)
    fld.coords['time'] = time - pd.to_timedelta(time.day-1, unit='d') + pd.DateOffset(days=14)

    return fld

# ----------------------------------------------------------------------------------------
def remove_annual_cycle(x, rec_per_year=12):
    '''
    Removes the annual cycle assuming that the first index is time. The optional arguemt
    rec_per_year gives the number of records per year. It defaults to 12
    '''

    if isinstance(x, xr.core.dataset.Dataset) or isinstance(x, xr.core.dataarray.DataArray):
        return (x.groupby('time.month') - x.groupby('time.month').mean(dim='time'))

    else:
        y = x.copy()

        for n in range(rec_per_year):
            y[n::rec_per_year,...] -= y[n::rec_per_year,...].mean(axis=0, keepdims=True)

        return y


##########################################################################################
# Spectra
##########################################################################################

# Maximum entropy spectral analysis
def memcoef(x, M):
    N = len(x)

    wk1 = x[:-1].copy()
    wk2 = np.roll(x, -1)[:-1]
    wkm = np.zeros(M)

    coef = np.zeros(M+1)

    for k in range(1, M+1):
        pneum = np.sum(wk1[:N-k]*wk2[:N-k])
        denom = np.sum(wk1[:N-k]**2 + wk2[:N-k]**2)

        coef[k] = 2*pneum/denom

        if k > 1:
            coef[1:k] = wkm[0:k-1] - coef[k]*wkm[k-2::-1]

        if k < M:
            wkm[:k] = coef[1:k+1]

            wk1_tmp = wk1.copy()
            wk1[0:N-k-1] -= wkm[k-1]*wk2[0:N-k-1]
            wk2[0:N-k-1] = wk2[1:N-k] - wkm[k-1]*wk1_tmp[1:N-k]

    coef[0] = x.var()*np.prod(1 - coef[1:]**2)

    return coef

# ----------------------------------------------------------------------------------------
def mempsd(f, b, dt):
    theta = 2j*np.pi*f*dt

    k = np.arange(1,len(b))[np.newaxis,:]
    zk = np.exp(theta[:,np.newaxis]*k)

    return b[0]/np.abs(1 - np.sum(b[1:]*zk, axis=1))**2


##########################################################################################
# Filtering
##########################################################################################

def pad_for_filter(x, padlen, padtype_left='const', padtype_right='const'):
    if padtype_left == 'const':
        x_left = np.full(padlen, x[0], dtype=np.array(x[0]).dtype)
    elif padtype_left == 'even':
        x_left = x[padlen:0:-1]
    elif padtype_left == 'odd':
        x_left = 2*x[0] - x[padlen:0:-1]
    else:
        raise RuntimeError('padtype_left must be \'const\', \'even\', or \'odd\'')

    if padtype_right == 'const':
        x_right = np.full(padlen, x[-1], dtype=np.array(x[-1]).dtype)
    elif padtype_right == 'even':
        x_right = x[-2:-padlen-2:-1]
    elif padtype_right == 'odd':
        x_right = 2*x[-1] - x[-2:-padlen-2:-1]
    else:
        raise RuntimeError('padtype_right must be \'const\', \'even\', or \'odd\'')

    return np.concatenate((x_left, x, x_right))

# ----------------------------------------------------------------------------------------
def lowpass(x, cutoff, order=3, fs=1, adapt_padtype=False, padtype_left='const', padtype_right='const',
           return_coefs=False):
    '''
    Note that the butterworth filter is scaled so the Nyquist frequency is 1 rather
    than scaling so that the sampling frequency is one.
    '''

    fn = fs/2
    b, a = sp.signal.butter(order, cutoff/fn)

    # padlen = 3*max(len(a),len(b))
    padlen = len(x)-2

    padtypes = ['const', 'even', 'odd']
    Npad = len(padtypes)

    rmse = np.zeros((Npad,Npad))
    if adapt_padtype is True:
        for i in range(Npad):
            for j in range(Npad):
                x_padded = pad_for_filter(x, padlen, padtype_left=padtypes[i], padtype_right=padtypes[j])
                y = sp.signal.filtfilt(b, a, x_padded, padtype=None)[padlen:-padlen]

                rmse[i,j] = np.var(y - x)

        i, j = np.unravel_index(np.argmin(rmse), (3,3))
        padtype_left = padtypes[i]
        padtype_right = padtypes[j]


    x_padded = pad_for_filter(x, padlen, padtype_left=padtype_left, padtype_right=padtype_right)
    y = sp.signal.filtfilt(b, a, x_padded, padtype=None)[padlen:-padlen]

    if adapt_padtype is False and return_coefs is False:
        return y
    else:
        retval = [y]

    if adapt_padtype is True:
        retval.append((padtype_left, padtype_right))

    if return_coefs is True:
        retval.append((b, a))

    return retval

# ----------------------------------------------------------------------------------------
def bandpass(x, cutoff, order=3, fs=1, adapt_padtype=False, padtype_left='const', padtype_right='const',
             return_coefs=False):

    fn = fs/2
    b, a = sp.signal.butter(order, [co/fn for co in cutoff], btype='bandpass')

    # padlen = 3*max(len(a),len(b))
    padlen = len(x)-2

    padtypes = ['const', 'even', 'odd']
    Npad = len(padtypes)

    rmse = np.zeros((Npad,Npad))
    if adapt_padtype is True:
        for i in range(Npad):
            for j in range(Npad):
                x_padded = pad_for_filter(x, padlen, padtype_left=padtypes[i], padtype_right=padtypes[j])
                y = sp.signal.filtfilt(b, a, x_padded, padtype=None)[padlen:-padlen]

                rmse[i,j] = np.var(y - x)

        i, j = np.unravel_index(np.argmin(rmse), (3,3))
        padtype_left = padtypes[i]
        padtype_right = padtypes[j]


    x_padded = pad_for_filter(x, padlen, padtype_left=padtype_left, padtype_right=padtype_right)
    y = sp.signal.filtfilt(b, a, x_padded, padtype=None)[padlen:-padlen]

    if adapt_padtype is False and return_coefs is False:
        return y
    else:
        retval = [y]

    if adapt_padtype is True:
        retval.append((padtype_left, padtype_right))

    if return_coefs is True:
        retval.append((b, a))

    return retval

##########################################################################################
# Correlations and covariances
##########################################################################################

def acorr(x, max_lag=None, detrend=False, pval=None, axis=0, biased=False, calc_pval=True):

    sz = list(x.shape)
    N = sz[axis]

    if max_lag is None:
        max_lag = N

    del sz[axis]
    sz = [max_lag] + sz

    r = np.ones(sz)
    p = np.zeros(sz)

    lags = np.arange(0, max_lag)

    if detrend:
        xp = standardize(sp.signal.detrend(x, axis=axis))
    else:
        xp = standardize(x)

    p[0] = np.NaN

    for n in range(1, len(lags)):
        r[n,...] = correlation(xp[lags[n]::,...], xp[:-lags[n]:,...], axis=axis, calc_pval=False)
        if biased:
            r[n,...] *= (N-lags[n])/N

    if calc_pval:
        if biased:
            ddof = N - 3
        else:
            ddof = N - 3 - lags

        if pval is None:

            ix = r < 1
            # z-score:
            z = np.zeros_like(r)
            p = np.zeros_like(r)

            z[ix] = np.sqrt(ddof)*np.arctanh(r[ix]) # z is distributed like a standard normal
            p[ix] = 2*(1 - sp.stats.norm.cdf(np.abs(z[ix])))

            return lags, r, p
        else:
            # return critical value
            r_crit = np.tanh(sp.stats.norm.ppf(1 - pval/2)/np.sqrt(ddof))

            return lags, r, r_crit

# ----------------------------------------------------------------------------------------
def acf(x):
    '''
    Autocorrelation function
    '''
    N = x.size

    x -= np.mean(x)

    r = np.correlate(x, x, mode='full')[(N-1):]
    r /= r[0]

    return r

# ----------------------------------------------------------------------------------------
def correlation(x, y, detrend=False, do_standardize=False, axis=0, calc_pval=True, pval=None):
    '''
    x should be 1D or the same dimension as y. y's 'axis' dimension should have the same length as x.

    If pval is a number, return critical r value rather than p value
    '''
    if detrend:
        xp = sp.signal.detrend(x, axis=axis)
        yp = sp.signal.detrend(y, axis=axis)
    else:
        xp = x.copy()
        yp = y.copy()

    if do_standardize:
        xp = standardize(xp, axis=axis)
        yp = standardize(yp, axis=axis)

    if x.ndim == 1:
        r = np.atleast_1d(np.nanmean(xp*np.rollaxis(yp, axis=0, start=yp.ndim), axis=-1))
    else:
        r = np.atleast_1d(np.nanmean(xp*yp, axis=axis))

    # catch cases where roundoff error makes r >= 1
    r[r > 1] = 1

    N = np.sum(~np.isnan(x), axis=axis)
    if calc_pval:
        if pval is None:

            ix = r < 1
            # z-score:
            z = np.zeros_like(r)
            p = np.zeros_like(r)

            z[ix] = np.sqrt(N-3)*np.arctanh(r[ix]) # z is distributed like a standard normal
            p[ix] = 2*(1 - sp.stats.norm.cdf(np.abs(z[ix])))

            return r, p
        else:
            # return critical value
            r_crit = np.tanh(sp.stats.norm.ppf(1 - pval/2)/np.sqrt(N-3))

            return r, r_crit
    else:
        return r

# ----------------------------------------------------------------------------------------
def corr_random_phase_fixed_nsamples(x, y, nsamples=500000, detrend=False):
    '''
    Correlation with random phase test for significance (Ebisuzaki, J. Climate 1997), fixed
    sample size version.

    Parameters
    ----------
    x, y : array_like
        Time series to correlate. Should be the same length.
    nsamples : int, optional
        Number of random samples. Default: 500,000.
    detrend : bool, optional
        Whether to detrend time series before correlating. Default: False.
    pval : float, optional
        If given, construct the 1 - p confidence intervals for coefficients

    Returns
    -------
    r : float
        Correlation coefficient.
    p : float
        p-value of correlation.

    Notes
    -----
    '''
    N = len(x)

    if detrend:
        x = sp.signal.detrend(x)
        y = sp.signal.detrend(y)

    x = standardize(x)
    y = standardize(y)

    X = np.fft.rfft(x)
    Y = np.fft.rfft(y)

    x_amp = np.abs(X)
    y_amp = np.abs(Y)

    X2 = np.repeat(x_amp[:,np.newaxis], nsamples, axis=1)
    Y2 = np.repeat(y_amp[:,np.newaxis], nsamples, axis=1)

    if np.mod(N, 2) == 1:
        r = 2*np.sum(np.real(X[1:]*np.conj(Y[1:])))/N**2
        r_sample = 2*np.sum(X2[1:,:]*Y2[1:,:]*
                            np.cos(2*pi*np.random.random_sample(size=(N//2, nsamples))), axis=0)/N**2
    else:
        r = (2*np.sum(np.real(X[1:-1]*np.conj(Y[1:-1]))) + np.real(X[-1]*Y[-1]))/N**2
        r_sample = 2*np.sum(X2[1:,:]*Y2[1:,:]*
                            np.cos(2*pi*np.random.random_sample(size=(N//2, nsamples))), axis=0)/N**2

    p = 1-np.sum(np.abs(r) > np.abs(r_sample))/nsamples

    return r, p

# ----------------------------------------------------------------------------------------
def corr_random_phase(x, y, detrend=False, axis=0, chunksize=10000, tol=1e-4, return_nsamples=False):
    '''
    Correlation with random phase test for significance (Ebisuzaki, J. Climate 1997). Adjusts
    sample size until p-value changes less than a tolerance.

    Parameters
    ----------
    x : 1D array
        "Primary" time series of length `NT`.
    y : array_like
        Time series to correlate with `x`. Can be multidimensional. Need `y.shape[axis] == NT`.
    detrend : bool, optional
        Which to detrend time series before correlation. Default: False.
    axis : int, optional
        Time axis of y. Default: 0
    chunksize : int, optional
        Perform chunksize number trials between checking tolerance. Default: 1e4.
    tol : float, optional
        Stop performing trials when performing `chunksize` new trials changes p-value by less than `tol`.
        Default: 1e-4
    return_nsamples : bool, optional
        If True, return the actual number of trials. Default: False

    Returns
    -------
    r : array-like
        Correlation coefficients.
    p : array-like
        p-values of correlations.
    nsamples : array-like
        Number of samples. Returned only if `return_nsamples = True`

    Notes
    -----
    '''

    @njit(parallel=True)
    def random_phase_multi(r, XY, NT, tol, chunksize):
        N = XY.shape[0]

        p_vec = np.zeros(N)
        nsamples = np.zeros(N)

        for n in prange(N):
            q = 0
            p = 0
            p_old = 1
            chunks = 0

            while np.abs(p - p_old) > tol:
                chunks += 1
                p_old = p


                for m in range(chunksize):
                    r_sample = 0
                    for nt in range(NT//2):
                        r_sample += 2*XY[n,nt]*np.cos(2*pi*np.random.random_sample())

                    r_sample /= NT**2

                    if np.abs(r[n]) > np.abs(r_sample):
                        q += 1

                p = 1 - q/(chunksize*chunks)

            p_vec[n] = p
            nsamples[n] = chunksize*chunks

        return p_vec, nsamples


    NT = len(x)
    y = np.moveaxis(y, axis, -1)

    if detrend:
        x = nandetrend(x)
        y = nandetrend(y, axis=-1)

    x = standardize(x)
    y = standardize(y, axis=-1)

    X = np.fft.rfft(x)
    Y = np.fft.rfft(y, axis=-1)

    x_amp = np.abs(X)
    y_amp = np.abs(Y)

    if np.mod(NT, 2) == 1:
        r = 2*np.sum(np.real(X[1:]*np.conj(Y[...,1:])), axis=-1)/NT**2
    else:
        r = (2*np.sum(np.real(X[1:-1]*np.conj(Y[...,1:-1])), axis=-1) + np.real(X[-1]*Y[...,-1]))/NT**2

    r = np.atleast_1d(r)
    XY = np.atleast_2d(x_amp[1:]*y_amp[...,1:])

    p = np.zeros(r.shape)
    nsamples = np.zeros(r.shape)

    idx = ~np.isnan(r)
    p[idx], nsamples[idx] = random_phase_multi(r[idx], XY[idx,:], NT, tol=tol, chunksize=chunksize)
    p[~idx] = np.nan


    if return_nsamples:
        return r, p, nsamples.astype(np.int)
    else:
        return r, p

# ----------------------------------------------------------------------------------------
def nandetrend(y, axis=0):
    '''
    Detrend in such a way that NaNs are handled gracefully.
    '''

    y = np.moveaxis(y, axis, -1)

    N = y.shape[-1]
    xp = np.arange(N) - (N-1)/2
    xvar = (N-1)*(N+1)/12

    yp = y - np.nanmean(y, axis=-1, keepdims=True)
    m = np.nanmean(xp*yp, axis=-1, keepdims=True)

    return np.moveaxis(yp - m*xp/xvar, -1, axis)

# ----------------------------------------------------------------------------------------
def xcorr(x, y, detrend=False, biased=False, pval=None, axis=0, calc_pval=True):
    '''
    x leads y for negative lags.
    '''

    sz = list(x.shape)
    N = sz[axis]

    del sz[axis]
    sz = [2*N+1] + sz

    r  = np.zeros(sz)
    p = np.zeros(sz)
    lags = np.arange(-N+1, N)

    if detrend:
        xp = standardize(sp.signal.detrend(x, axis=axis))
        yp = standardize(sp.signal.detrend(y, axis=axis))
    else:
        xp = standardize(x)
        yp = standardize(y)

    if biased:
        dof = N*np.ones_like(lags)
    else:
        dof = N - np.abs(lags)

    r = np.correlate(xp, yp, mode='full')/dof

    if calc_pval:
        if pval is None:

            ix = r < 1
            # z-score:
            z = np.zeros_like(r)
            p = np.zeros_like(r)

            z[ix] = np.sqrt(dof-3)*np.arctanh(r[ix]) # z is distributed like a standard normal
            p[ix] = 2*(1 - sp.stats.norm.cdf(np.abs(z[ix])))

            return lags, r, p
        else:
            # return critical value
            ix = dof - 3 > 0
            r_crit = np.ones_like(r)
            r_crit[ix] = np.tanh(sp.stats.norm.ppf(1 - pval/2)/np.sqrt(dof[ix]-3))

            return lags, r, r_crit
    else:
        return lags, r

# ----------------------------------------------------------------------------------------
def lag_correlation(x, y, max_lag=5, detrend=False, pval=None, axis=0):
    return xcorr(x, y, max_lag, detrend, pval, axis)

# ----------------------------------------------------------------------------------------
def cov2(x, y, ddof=0, axis=0):
    '''
    Covariance matrix between two N-dimensional time series
    '''

    sz = list(x.shape)
    N = sz[axis]
    del sz[axis]

    S = np.zeros([2,2] + sz)

    xp = x - x.mean(axis=axis, keepdims=True)
    yp = y - y.mean(axis=axis, keepdims=True)

    S[0,0,...] = np.sum(xp**2, axis=axis)/(N - ddof)
    S[0,1,...] = np.sum(xp*yp, axis=axis)/(N - ddof)
    S[1,0,...] = S[0,1,...]
    S[1,1,...] = np.sum(yp**2, axis=axis)/(N - ddof)

    return S


##########################################################################################
# Partial correlations and covariances
##########################################################################################

def pacorr(x, max_lag=10, pval=None, detrend=True):
    '''
    Partial autocorrelation function.

    returns lags, partial autocorrelation and
        p-value if pval is None
        critical value if pval is not None
    '''
    pacf = np.zeros((max_lag+1, max_lag+1))

    N = len(x)
    lags, acf, _ = acorr(x, max_lag=max_lag, detrend=detrend)

    pacf[1,1] = acf[1]
    for p in range(2, max_lag+1):
        pacf[p, p] = (acf[p] - np.sum(pacf[p-1, 1:p]*acf[p-1:0:-1]))/(1 - np.sum(pacf[p-1, p-1:0:-1]*acf[p-1:0:-1]))
        for k in range(1, p):
            pacf[p, k] = pacf[p-1, k] - pacf[p, p]*pacf[p-1, p-k]

    pacf = np.diag(pacf[1:,1:])
    if pval is None:
        # z-score:
        z = np.sqrt(N)*pacf # z is distributed like a standard normal
        p = 2*(1 - sp.stats.norm.cdf(np.abs(z)))

        return lags[1:], pacf, p
    else:
        # return critical value
        a_crit = sp.stats.norm.ppf(1 - pval/2)/np.sqrt(N)

        return lags[1:], pacf, a_crit

# ----------------------------------------------------------------------------------------
def partial_correlation(x, y, z, detrend=False, axis=0, pval=None):
    '''
    partial_correlation(x, y, z)

    Partial correlation of x and y holding z fixed.
    '''

    if detrend:
        xp = sp.signal.detrend(x, axis=axis)
        yp = sp.signal.detrend(y, axis=axis)
        zp = sp.signal.detrend(z, axis=axis)
    else:
        xp = x
        yp = y
        zp = z

    rxy, _ = correlation(xp, yp, axis=axis, detrend=False)
    rxz, _ = correlation(zp, xp, axis=axis, detrend=False)
    ryz, _ = correlation(zp, yp, axis=axis, detrend=False)

    r = (rxy - rxz*ryz)/(np.sqrt(1 - rxz**2)*np.sqrt(1 - ryz**2))

    N = x.shape[0]
    if pval is None:
        # z-score:
        z = np.sqrt(N-4)*np.arctanh(r) # z is distributed like a standard normal
        p = 2*(1 - sp.stats.norm.cdf(np.abs(z)))

        return r, p
    else:
        # return critical value
        r_crit = np.tanh(sp.stats.norm.ppf(1 - pval/2)/np.sqrt(N-4))

        return r, r_crit

# ----------------------------------------------------------------------------------------
def partial_correlation_1d(x, y, z, detrend=False, pval=None):
    '''
    partial_correlation_1d(x, y, z)

    Partial correlation of x and y holding z fixed. x and y are assumed to be 1D with
    time as the first axis. z can be N x K, where N is the length of x and y.
    '''

    if detrend:
        xp = sp.signal.detrend(x, axis=0)
        yp = sp.signal.detrend(y, axis=0)
        zp = sp.signal.detrend(z, axis=0)
    else:
        xp = x
        yp = y
        zp = z

#     from IPython.core.debugger import Tracer
#     Tracer()()

    if zp.ndim > 1:
        N, K = zp.shape
        model = np.ones((N, K+1))
        model[:,1:] = zp
    else:
        N = len(zp)
        K = 1
        model = np.ones((N, 2))
        model[:,1] = zp

    # perform regressions to find the residuals
    _, x_regress, _ = regress(x, model)
    _, y_regress, _ = regress(y, model)

    x_res = standardize(xp - x_regress)
    y_res = standardize(yp - y_regress)

    r = np.mean(x_res*y_res)

    if pval is None:
        # z-score:
        z = np.sqrt(N-K-3)*np.arctanh(r) # z is distributed like a standard normal
        p = 2*(1 - sp.stats.norm.cdf(np.abs(z)))

        return r, p
    else:
        # return critical value
        r_crit = np.tanh(sp.stats.norm.ppf(1 - pval/2)/np.sqrt(N-K-3))

        return r, r_crit

# ----------------------------------------------------------------------------------------
def partial_covariance_1d(x, y, z, detrend=False, biased=False, normalize=True):
    '''
    partial_covariance_1d(x, y, z)

    Partial correlation of x and y holding z fixed. x and y are assumed to be 1D with
    time as the first axis. z can be N x K, where N is the length of x and y.
    '''

    if detrend:
        xp = sp.signal.detrend(x, axis=0)
        yp = sp.signal.detrend(y, axis=0)
        zp = sp.signal.detrend(z, axis=0)
    else:
        xp = x
        yp = y
        zp = z

#     from IPython.core.debugger import Tracer
#     Tracer()()

    if zp.ndim > 1:
        N, K = zp.shape
        model = np.ones((N, K+1))
        model[:,1:] = zp
    else:
        N = len(zp)
        K = 1
        model = np.ones((N, 2))
        model[:,1] = zp

    # perform regressions to find the residuals
    _, x_regress, _ = regress(x, model)
    _, y_regress, _ = regress(y, model)

    x_res = xp - x_regress
    y_res = yp - y_regress

    r = np.sum(x_res*y_res)
    if normalize:
        r /= N

    return r

# ----------------------------------------------------------------------------------------
def partial_xcorr(x, y, z, max_lag=5, detrend=False, pval=None):
    N = len(x)
    r  = np.zeros(2*max_lag+1)
    p = np.zeros(2*max_lag+1)
    lags = np.arange(-max_lag, max_lag+1)

#     from IPython.core.debugger import Tracer
#     Tracer()()

    if detrend:
        xp = sp.signal.detrend(x, axis=0)
        yp = sp.signal.detrend(y, axis=0)
        zp = sp.signal.detrend(z, axis=0)
    else:
        xp = x
        yp = y
        zp = z

    for n, lag in enumerate(lags):
        if lag == 0:
            r[n], p[n] = partial_correlation_1d(xp, yp, zp, pval=pval)
        elif lag > 0:
            r[n], p[n] = partial_correlation_1d(xp[lag::], yp[:-lag:], zp[:-lag:,...], pval=pval)
#             r[n] *= (N-lag)/N
        else:
            r[n], p[n] = partial_correlation_1d(xp[:lag:], yp[-lag::], zp[-lag::,...], pval=pval)
#             r[n] *= (N+lag)/N

    return lags, r, p

# ----------------------------------------------------------------------------------------
def lag_partial_correlation(x, y, max_lag=5, detrend=False, pval=None, axis=0):
    return partial_xcorr(x, y, mag_lag, detrend, pval, axis)


##########################################################################################
# Regression
##########################################################################################

def regress(data, model, error=None, cutoff=1e-7, verbose=False, pval=None):
    '''
    Least squares regression with a check for singular components.

    Parameters
    ----------
    data : array_like, length N
        Time series of data to fit.
    model : array_like, shape N x K
        Array of K predictors evaluated at the N measurement times/locations.
    error : array_like, length N, optional
        Standard errors of each measurement in 'data'. If not specified or None, data
        is not weighted by error estimate.
    cutoff : float, optional
        Models with singular values less than the maximum singular value by a factor of
        more than 'cutoff' are set to zero. Default: 1e-7.
    verbose : bool, optional
        If true, print fit statistics. Default: False.
    pval : float, optional
        If given, construct the 1 - p confidence intervals for coefficients

    Returns
    -------
    coefs : array_like, length K
        Coefficients of regression.
    x_regress : array_like, length N
        Regression evaluated with 'model' input.
    stats : dict
        Statistics of the regression:
            ssr: sum of squares due to regression
            sse: sum of squares of error
            sst: total sum of squares
            Fstat: f-statistic
            pval: p-value of regression
            S: singular values of design matrix
            Cov: covariance of coefficients

    Notes
    -----
    '''
    import numpy.ma as ma
    from scipy.linalg import svd
    from scipy.stats import f, t


    # multplication of masked arrays is broken in numpy 1.17
    if isinstance(data, ma.MaskedArray):
        input_is_ma = True
        data_mask = data.mask[0].copy()
        data = data.data
    else:
        input_is_ma = False

    if model.ndim == 1:
        model = model[:,np.newaxis]
    N, K = model.shape

    dfR = K
    dfE = N - K

    if error is not None:
        raise RuntimeError('Weighed regression not implmented yet')

    # convert data into a row matrix
    reshape_output = False
    if data.ndim == 1:
        data = data[:,np.newaxis]
    elif data.shape[1] > 1 or data.ndim > 2:
        data_shape = data.shape
        data = data.reshape((data_shape[0], np.prod(data_shape[1:])))
        reshape_output = True
        verbose = False

    # Design matrix (von Storch and Zwiers 8.3.19)
    X = model
    y = data

    U, S, Vt = svd(X, full_matrices=False)

    Sinv = np.zeros_like(S)
    idx = S/S.max() > cutoff
    if np.any(idx == 0):
        warnings.warn('Models are not linearly independent.')
    Sinv[idx] = 1/S[idx]
    Si = np.diag(Sinv)
#     breakpoint()
    coefs = Vt.T @ Si @ U.T @ y

    y_regress = model @ coefs
    sse = np.sum((y - y_regress)**2, axis=0)
    sst = np.sum((y - y.mean(axis=0))**2, axis=0)
    ssr = sst - sse

    stats = {'dfR': dfR, 'dfE': dfE, 'ssr': ssr, 'sse': sse, 'sst': sst, 'S': S, 'rSqr': ssr/sst}

    # Variance of the error and covariance of coefficients (von Storch and Zwiers 8.4.2)
    sigE = np.sqrt(sse/dfE)
    weights = Vt.T @ Si**2 @ Vt
    Cov = np.squeeze(weights[:,:,np.newaxis]*sigE**2)

    stats['sigE'] = sigE
    stats['Cov'] = Cov

    # test whether a regression relationship exists (von Storch and Zwiers 8.4.8)
    Fstat = (ssr/sse)*(dfE/dfR)
    pval_regress = 1-f.cdf(Fstat, dfR, dfE)

    stats['Fstat'] = Fstat
    stats['pval_regress'] = pval_regress

    # confidence intervals for coefficients (von Storch and Zwiers 8.4.6)
    if pval is not None:
        coef_ci = t.ppf(1-pval/2, dfE)*np.sqrt(np.diag(weights)[:,np.newaxis])*sigE
        stats['coef_ci'] = coef_ci

#     import pdb; pdb.set_trace()
    # p values for the coefficients (inversion of von Storch and Zwiers 8.4.6)
    pval_coef = 2*(1 - t.cdf(np.abs(coefs)/(np.sqrt(np.diag(weights)[:,np.newaxis])*sigE), dfE))
    stats['pval_coef'] = pval_coef

    if verbose:
        print('')
        print('source | sum of sq.    df')
        print('-------------------------')
        print('ssr    |    {:5.1f}      {:2d}'.format(ssr[0], K))
        print('sse    |    {:5.1f}      {:2d}'.format(sse[0], N-K))
        print('sst    |    {:5.1f}      {:2d}'.format(sst[0], N))
        print('')
        print('Ratio of smallest to largest singular value: {:g}'.format(S[-1]/S[0]))
        print('Fstat: {:.4f}, p-val: {:.2g}'.format(Fstat[0], pval_regress[0]))
        print('Variance explained: {:.3%}'.format(ssr[0]/sst[0]))
        print('')

    if reshape_output:
        coefs = coefs.reshape((K,) + data_shape[1:])
        y_regress = y_regress.reshape((N,) + data_shape[1:])

        for key in ('ssr', 'sse', 'sst', 'rSqr', 'sigE', 'Fstat', 'pval_regress'):
            stats[key] = np.reshape(stats[key], data_shape[1:])

        try:
            stats['coef_ci'] = np.reshape(stats['coef_ci'], (K,) + data_shape[1:])
        except:
            pass
        stats['pval_coef'] = np.reshape(stats['pval_coef'], (K,) + data_shape[1:])
        stats['Cov'] = np.reshape(stats['Cov'], (K, K) + data_shape[1:])

    if input_is_ma: # restore mask
        coefs = coefs.view(ma.MaskedArray)
        coefs[:, data_mask] = ma.masked

        y_regress = y_regress.view(ma.MaskedArray)
        y_regress[:, data_mask] = ma.masked

        for key in ('ssr', 'sse', 'sst', 'rSqr', 'sigE', 'Fstat', 'pval_regress'):
            stats[key] = stats[key].view(ma.MaskedArray)
            stats[key][data_mask] = ma.masked

        stats['pval_coef'] = stats['pval_coef'].view(ma.MaskedArray)
        stats['pval_coef'][:,data_mask] = ma.masked

    return coefs, y_regress, stats
##########################################################################################
# Statistical tests
##########################################################################################

def ttest(x1, x2, axis=0, dim=None, return_statistics=False):
    '''
    Univariate t-test for significance of differences of means based on Welch's T test.

    Parameters
    ----------
    x1, x2 : array_like or xarray
        Sets to test. If multidimensional, must have same dimensions along extra axes.
    axis : int, optional (for array_like input)
        Axis along which to perform test. Default: 0.
    dim : str (for xarray input)
        Dimension along which to perform test.
    return_statistics : bool, optional
        If true, statistics are returned as well as p-value

    Returns
    -------
    pval : float
        p-value
    stats : dict, only if return_statistic == True
        Test statistics:
            tstat: t-statistic
            N1: number of samples in first set
            N2: number of samples in second set
            df: effective degrees of freedom

    Notes
    -----
    '''

    dx = x1.mean(axis=axis) - x2.mean(axis=axis)

    var1 = x1.var(axis=axis, ddof=1)
    var2 = x2.var(axis=axis, ddof=1)

    N1 = x1.shape[axis]
    N2 = x2.shape[axis]

    tstat = dx/np.sqrt(var1/N1 + var2/N2)

    nu1 = N1 - 1
    nu2 = N2 - 1

    df = np.floor((var1/N1 + var2/N2)**2
                  / ((var1/N1)**2/nu1 + (var2/N2)**2/nu2))

    pval = 2*(1-sp.stats.t.cdf(np.abs(tstat), df))

    stats = {'tstat': tstat, 'N1': N1, 'N2': N2, 'df': df}
    if return_statistics:
        return pval, stats
    else:
        return pval

# ----------------------------------------------------------------------------------------
def ttest_bivariate(x1, x2, y1, y2, axis=0, return_statistic=False):
    '''
    Bivariate t-test for significance of differences of means based on Hotellings
    T-squared test.

    Parameters
    ----------
    x1, x2 : array_like
        Sets of first varianble to test. If multidimensional, must have same dimensions
        along extra axes.
    y1, y2 : array_like, can be multidimensional
        Sets of second varianble to test. If multidimensional, must have same dimensions
        as x1 and x2 along extra axes.
    axis : int, optional
        Axis along which to perform test. Default: 0.
    return_statistic : bool, optional
        If true, f-statistic is returned as well as p-value

    Returns
    -------
    pval : float
        p-value
    fstat : float, only if return_statistic == True
        f-statistic

    Notes
    -----
    '''

    dx = x1.mean(axis=axis) - x2.mean(axis=axis)
    dy = y1.mean(axis=axis) - y2.mean(axis=axis)

    S1 = cov2(x1, y1, ddof=1, axis=axis)
    S2 = cov2(x2, y2, ddof=1, axis=axis)

    N1 = x1.shape[axis]
    N2 = x2.shape[axis]

    # pooled variance
    Sp = ((N1 - 1)*S1 + (N2-1)*S2)/(N1 + N2 - 2)

    t2stat = ((N1*N2/(N1 + N2))
                * (Sp[1,1]*dx**2 - 2*Sp[0,1]*dx*dy + Sp[0,0]*dy**2)
                / (Sp[0,0]*Sp[1,1] - Sp[0,1]**2))

    fstat = (N1 + N2 - 3)*t2stat/((N1 + N2 - 2)*2)

    df1 = 2
    df2 = N1 + N2 - 3

    pval = 1-sp.stats.f.cdf(fstat, df1, df2)

    if return_statistic:
        return pval, fstat
    else:
        return pval
