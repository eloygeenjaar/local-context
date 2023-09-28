import yaml
import torch
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr
from scipy.optimize import linear_sum_assignment


def positive_sample(reference_ix: np.ndarray, max_ix: int,
                    delta: int) -> int:
    num_references = reference_ix.shape[0]
    deltas = np.array([delta] * num_references)
    left_bound_arr = np.stack((-deltas, -reference_ix), axis=1)
    left_bound = np.max(left_bound_arr, axis=1)
    right_bound_arr = np.stack((deltas, max_ix - reference_ix), axis=1)
    right_bound = np.min(right_bound_arr, axis=1)
    positive_ix = np.random.randint(left_bound, right_bound)
    return reference_ix + positive_ix

def create_batches(num_timesteps: int, num_timesteps_batch: int):
    timesteps = np.arange(num_timesteps)
    timestep_permutation = np.random.permutation(timesteps)
    # Not all batches willl be the exact same size
    batches = np.array_split(timestep_permutation, num_timesteps // num_timesteps_batch + 1)
    return batches

def auction_linear_assignment(x, eps=None, reduce='sum'):
    """
    Solve the linear sum assignment problem using the auction algorithm.
    Implementation in pytorch, GPU compatible.

    x_ij is the affinity between row (person) i and column (object) j, the
    algorithm aims to assign to each row i a column j_i such that the total benefit
    \sum_i x_{ij_i} is maximized.

    pytorch implementation, supports GPU.

    Algorithm adapted from http://web.mit.edu/dimitrib/www/Auction_Survey.pdf

    :param x: torch.Tensor
            The affinity (or benefit) matrix of size (n, n)
    :param eps: float, optional
            Bid size. Smaller values yield higher accuracy at the price of
            longer runtime.
    :param reduce: str, optional
            The reduction method to be applied to the score.
            If `sum`, sum the entries of cost matrix after assignment.
            If `mean`, compute the mean of the cost matrix after assignment.
            If `none`, return the vector (n,) of assigned column entry per row.
    :return: (torch.Tensor, torch.Tensor, int)
            Tuple of (score after application of reduction method, assignment,
            number of steps in the auction algorithm).
    """
    eps = 1 / x.size(0) if eps is None else eps

    price = torch.zeros((1, x.size(1))).to(x.device)
    assignment = torch.zeros(x.size(0)).long().to(x.device) - 1
    bids = torch.zeros_like(x).to(x.device)

    n_iter = 0
    while (assignment == -1).any():
        n_iter += 1

        # -- Bidding --
        # set I of unassigned rows (persons)
        # a person is unassigned if it is assigned to -1
        I = (assignment == -1).nonzero().squeeze(dim=1)
        # value matrix = affinity - price
        value_I = x[I, :] - price
        # find j_i, the best value v_i and second best value w_i for each i \in I
        top_value, top_idx = value_I.topk(2, dim=1)
        jI = top_idx[:, 0]
        vI, wI = top_value[:, 0], top_value[:, 1]
        # compute bid increments \gamma
        gamma_I = vI - wI + eps
        # fill entry (i, j_i) with \gamma_i for each i \in I
        # every unassigned row i makes a bid at one j_i with value \gamma_i
        bids_ = bids[I, :]
        bids_.zero_()
        bids_.scatter_(dim=1, index=jI.contiguous().view(-1, 1), src=gamma_I.view(-1, 1))

        # -- Assignment --
        # set J of columns (objects) that have at least a bidder
        # if a column j in bids_ is empty, then no bid was made to object j
        J = (bids_ > 0).sum(dim=0).nonzero().squeeze(dim=1)
        # determine the highest bidder i_j and corresponding highest bid \gamma_{i_j}
        # for each object j \in J
        gamma_iJ, iJ = bids_[:, J].max(dim=0)
        # since iJ is the index of highest bidder in the "smaller" array bids_,
        # find its actual index among the unassigned rows I
        # now iJ is a subset of I
        iJ = I[iJ]
        # raise the price of column j by \gamma_{i_j} for each j \in J
        price[:, J] += gamma_iJ
        # unassign any row that was assigned to object j at the beginning of the iteration
        # for each j \in J
        mask = (assignment.view(-1, 1) == J.view(1, -1)).sum(dim=1).byte()
        assignment.masked_fill_(mask, -1)
        # assign j to i_j for each j \in J
        assignment[iJ] = J

    score = x.gather(dim=1, index=assignment.view(-1, 1)).squeeze()
    if reduce == 'sum':
        score = torch.sum(score)
    elif reduce == 'mean':
        score = torch.mean(score)
    elif reduce == 'none':
        pass
    else:
        raise ValueError('not a valid reduction method: {}'.format(reduce))

    return score, assignment, n_iter


def rankdata_pt(b, tie_method='ordinal', dim=0):
    """
    pytorch equivalent of scipy.stats.rankdata, GPU compatible.

    :param b: torch.Tensor
            The 1-D or 2-D tensor of values to be ranked. The tensor is first flattened
            if tie_method is not 'ordinal'.
    :param tie_method: str, optional
            The method used to assign ranks to tied elements.
                The options are 'average', 'min', 'max', 'dense' and 'ordinal'.
                'average':
                    The average of the ranks that would have been assigned to
                    all the tied values is assigned to each value.
                    Supports 1-D tensors only.
                'min':
                    The minimum of the ranks that would have been assigned to all
                    the tied values is assigned to each value.  (This is also
                    referred to as "competition" ranking.)
                    Supports 1-D tensors only.
                'max':
                    The maximum of the ranks that would have been assigned to all
                    the tied values is assigned to each value.
                    Supports 1-D tensors only.
                'dense':
                    Like 'min', but the rank of the next highest element is assigned
                    the rank immediately after those assigned to the tied elements.
                    Supports 1-D tensors only.
                'ordinal':
                    All values are given a distinct rank, corresponding to the order
                    that the values occur in `a`.
                The default is 'ordinal' to match argsort.
    :param dim: int, optional
            The axis of the observation in the data if the input is 2-D.
            The default is 0.
    :return: torch.Tensor
            An array of length equal to the size of `b`, containing rank scores.
    """
    # b = torch.flatten(b)

    if b.dim() > 2:
        raise ValueError('input has more than 2 dimensions')
    if b.dim() < 1:
        raise ValueError('input has less than 1 dimension')

    order = torch.argsort(b, dim=dim)

    if tie_method == 'ordinal':
        ranks = order + 1
    else:
        if b.dim() != 1:
            raise NotImplementedError('tie_method {} not supported for 2-D tensors'.format(tie_method))
        else:
            n = b.size(0)
            ranks = torch.empty(n).to(b.device)

            dupcount = 0
            total_tie_count = 0
            for i in range(n):
                inext = i + 1
                if i == n - 1 or b[order[i]] != b[order[inext]]:
                    if tie_method == 'average':
                        tie_rank = inext - 0.5 * dupcount
                    elif tie_method == 'min':
                        tie_rank = inext - dupcount
                    elif tie_method == 'max':
                        tie_rank = inext
                    elif tie_method == 'dense':
                        tie_rank = inext - dupcount - total_tie_count
                        total_tie_count += dupcount
                    else:
                        raise ValueError('not a valid tie_method: {}'.format(tie_method))
                    for j in range(i - dupcount, inext):
                        ranks[order[j]] = tie_rank
                    dupcount = 0
                else:
                    dupcount += 1
    return ranks


def cov_pt(x, y=None, rowvar=False):
    """
    Estimate a covariance matrix given data in pytorch, GPU compatible.

    Covariance indicates the level to which two variables vary together.
    If we examine N-dimensional samples, `X = [x_1, x_2, ... x_N]^T`,
    then the covariance matrix element `C_{ij}` is the covariance of
    `x_i` and `x_j`. The element `C_{ii}` is the variance of `x_i`.

    :param x: torch.Tensor
            A 1-D or 2-D array containing multiple variables and observations.
            Each column of `x` represents a variable, and each row a single
            observation of all those variables.
    :param y: torch.Tensor, optional
            An additional set of variables and observations. `y` has the same form
            as that of `x`.
    :param rowvar: bool, optional
            If `rowvar` is True, then each row represents a
            variable, with observations in the columns. Otherwise, the
            relationship is transposed: each column represents a variable,
            while the rows contain observations.
            The default is False.
    :return: torch.Tensor
            The covariance matrix of the variables.
    """
    if y is not None:
        if not x.size() == y.size():
            raise ValueError('x and y have different shapes')
    if x.dim() > 2:
        raise ValueError('x has more than 2 dimensions')
    if x.dim() < 2:
        x = x.view(1, -1)
    if not rowvar and x.size(0) != 1:
        x = x.t()
    if y is not None:
        if y.dim() < 2:
            y = y.view(1, -1)
        if not rowvar and y.size(0) != 1:
            y = y.t()
        x = torch.cat((x, y), dim=0)

    fact = 1.0 / (x.size(1) - 1)
    x -= torch.mean(x, dim=1, keepdim=True)
    xt = x.t()  # if complex: xt = x.t().conj()
    return fact * x.matmul(xt).squeeze()


def corrcoef_pt(x, y=None, rowvar=False):
    """
    Return Pearson product-moment correlation coefficients in pytorch, GPU compatible.

    Implementation very similar to numpy.corrcoef using cov.

    :param x: torch.Tensor
            A 1-D or 2-D array containing multiple variables and observations.
            Each row of `m` represents a variable, and each column a single
            observation of all those variables.
    :param y: torch.Tensor, optional
            An additional set of variables and observations. `y` has the same form
            as that of `m`.
    :param rowvar: bool, optional
            If `rowvar` is True, then each row represents a
            variable, with observations in the columns. Otherwise, the
            relationship is transposed: each column represents a variable,
            while the rows contain observations.
            The default is False.
    :return: torch.Tensor
            The correlation coefficient matrix of the variables.
    """
    c = cov_pt(x, y, rowvar)
    try:
        d = torch.diag(c)
    except RuntimeError:
        # scalar covariance
        return c / c
    stddev = torch.sqrt(d)
    c /= stddev[:, None]
    c /= stddev[None, :]

    return c


def spearmanr_pt(x, y=None, rowvar=False):
    """
    Calculates a Spearman rank-order correlation coefficient in pytorch, GPU compatible.

    :param x: torch.Tensor
            A 1-D or 2-D array containing multiple variables and observations.
            Each column of `x` represents a variable, and each row a single
            observation of all those variables.
    :param y: torch.Tensor, optional
            An additional set of variables and observations. `y` has the same form
            as that of `x`.
    :param rowvar: bool, optional
            If `rowvar` is True, then each row represents a
            variable, with observations in the columns. Otherwise, the
            relationship is transposed: each column represents a variable,
            while the rows contain observations.
            The default is False.
    :return: torch.Tensor
           Spearman correlation matrix or correlation coefficient.
    """
    xr = rankdata_pt(x, dim=int(rowvar)).float()
    yr = None
    if y is not None:
        yr = rankdata_pt(y, dim=int(rowvar)).float()
    rs = corrcoef_pt(xr, yr, rowvar)
    return rs


def mean_corr_coef_pt(x, y, method='pearson'):
    """
    A differentiable pytorch implementation of the mean correlation coefficient metric.

    :param x: torch.Tensor
    :param y: torch.Tensor
    :param method: str, optional
            The method used to compute the correlation coefficients.
                The options are 'pearson' and 'spearman'
                'pearson':
                    use Pearson's correlation coefficient
                'spearman':
                    use Spearman's nonparametric rank correlation coefficient
    :return: float
    """
    d = x.size(1)
    if method == 'pearson':
        cc = corrcoef_pt(x, y)[:d, d:]
    elif method == 'spearman':
        cc = spearmanr_pt(x, y)[:d, d:]
    else:
        raise ValueError('not a valid method: {}'.format(method))
    cc = torch.abs(cc)
    score, _, _ = auction_linear_assignment(cc, reduce='mean')
    return score


def mean_corr_coef_np(x, y, method='pearson'):
    """
    A numpy implementation of the mean correlation coefficient metric.

    :param x: numpy.ndarray
    :param y: numpy.ndarray
    :param method: str, optional
            The method used to compute the correlation coefficients.
                The options are 'pearson' and 'spearman'
                'pearson':
                    use Pearson's correlation coefficient
                'spearman':
                    use Spearman's nonparametric rank correlation coefficient
    :return: float
    """
    d = x.shape[1]
    if method == 'pearson':
        cc = np.corrcoef(x, y, rowvar=False)[:d, d:]
    elif method == 'spearman':
        cc = spearmanr(x, y)[0][:d, d:]
    else:
        raise ValueError('not a valid method: {}'.format(method))
    cc = np.abs(cc)
    score = cc[linear_sum_assignment(-1 * cc)].mean()
    return score

def mean_corr_coef(x, y, method='pearson'):
    if type(x) != type(y):
        raise ValueError('inputs are of different types: ({}, {})'.format(type(x), type(y)))
    if isinstance(x, np.ndarray):
        return mean_corr_coef_np(x, y, method)
    elif isinstance(x, torch.Tensor):
        return mean_corr_coef_pt(x, y, method)
    else:
        raise ValueError('not a supported input type: {}'.format(type(x)))

def get_default_config(args):
    if len(args) > 1:
        with Path(f'configs/config_{int(args[1])}.yaml').open('r') as f:
            default_conf = yaml.safe_load(f)
    else:
        with Path('configs/default.yaml').open('r') as f:
            default_conf = yaml.safe_load(f)
    return dict(default_conf)

# Adapted from: https://github.com/googleinterns/local_global_ts_representation/blob/main/gl_rep/data_loaders.py
def get_physionet(dataset = 'set-a'):
    """Function to load the The PhysioNet Computing in Cardiology Challenge 2012 dataset into TF dataset objects

    The data is loaded, normalized, padded, and a mask channel is generated to indicate missing observations
    The raw csv files can be downloaded from:
    https://physionet.org/content/challenge-2012/1.0.0/
    A number of steps are borrowed from this repo: https://github.com/alistairewj/challenge2012

    Args:
        normalize: The type of data normalizatino to perform ["none", "mean_zero", "min_max"]
    """
    feature_map = {'Albumin': 'Serum Albumin (g/dL)',
                'ALP': 'Alkaline phosphatase (IU/L)',
                'ALT': 'Alanine transaminase (IU/L)',
                'AST': 'Aspartate transaminase (IU/L)',
                'Bilirubin': 'Bilirubin (mg/dL)',
                'BUN': 'Blood urea nitrogen (mg/dL)',
                'Cholesterol': 'Cholesterol (mg/dL)',
                'Creatinine': 'Serum creatinine (mg/dL)',
                'DiasABP': 'Invasive diastolic arterial blood pressure (mmHg)',
                'FiO2': 'Fractional inspired O2 (0-1)',
                'GCS': 'Glasgow Coma Score (3-15)',
                'Glucose': 'Serum glucose (mg/dL)',
                'HCO3': 'Serum bicarbonate (mmol/L)',
                'HCT': 'Hematocrit (%)',
                'HR': 'Heart rate (bpm)',
                'K': 'Serum potassium (mEq/L)',
                'Lactate': 'Lactate (mmol/L)',
                'Mg': 'Serum magnesium (mmol/L)',
                'MAP': 'Invasive mean arterial blood pressure (mmHg)',
                'Na': 'Serum sodium (mEq/L)',
                'NIDiasABP': 'Non-invasive diastolic arterial blood pressure (mmHg)',
                'NIMAP': 'Non-invasive mean arterial blood pressure (mmHg)',
                'NISysABP': 'Non-invasive systolic arterial blood pressure (mmHg)',
                'PaCO2': 'partial pressure of arterial CO2 (mmHg)',
                'PaO2': 'Partial pressure of arterial O2 (mmHg)',
                'pH': 'Arterial pH (0-14)',
                'Platelets': 'Platelets (cells/nL)',
                'RespRate': 'Respiration rate (bpm)',
                'SaO2': 'O2 saturation in hemoglobin (%)',
                'SysABP': 'Invasive systolic arterial blood pressure (mmHg)',
                'Temp': 'Temperature (°C)',
                'TroponinI': 'Troponin-I (μg/L)',
                'TroponinT': 'Troponin-T (μg/L)',
                'Urine': 'Urine output (mL)',
                'WBC': 'White blood cell count (cells/nL)'
                   }
    feature_list = list(feature_map.keys())
    local_list = ['MechVent', 'Weight']
    data_dir = './data/physionet'
    static_vars = ['RecordID', 'Age', 'Gender', 'Height', 'ICUType', 'Weight']
    
    if os.path.exists(('./data/physionet/processed_df.csv')):
        df_full = pd.read_csv('./data/physionet/processed_df.csv')
        df_static = pd.read_csv('./data/physionet/processed_static_df.csv')
    else:
        txt_all = list()
        for f in os.listdir(os.path.join(data_dir, dataset)):
            with open(os.path.join(data_dir, dataset, f), 'r') as fp:
                txt = fp.readlines()
            # get recordid to add as a column
            recordid = txt[1].rstrip('\n').split(',')[-1]
            try:
                txt = [t.rstrip('\n').split(',') + [int(recordid)] for t in txt]
                txt_all.extend(txt[1:])
            except:
                continue

        # convert to pandas dataframe
        df = pd.DataFrame(txt_all, columns=['time', 'parameter', 'value', 'recordid'])

        # extract static variables into a separate dataframe
        df_static = df.loc[df['time'] == '00:00', :].copy()

        df_static = df_static.loc[df['parameter'].isin(static_vars)]

        # remove these from original df
        idxDrop = df_static.index
        df = df.loc[~df.index.isin(idxDrop), :]

        # pivot on parameter so there is one column per parameter
        df_static = df_static.pivot(index='recordid', columns='parameter', values='value')

        # some conversions on columns for convenience
        df['value'] = pd.to_numeric(df['value'], errors='raise')
        df['time'] = df['time'].map(lambda x: int(x.split(':')[0]) * 60 + int(x.split(':')[1]))

        df.head()
        # convert static into numeric
        for c in df_static.columns:
            df_static[c] = pd.to_numeric(df_static[c])

        # preprocess
        for c in df_static.columns:
            x = df_static[c]
            if c == 'Age':
                # replace anon ages with 91.4
                idx = x > 130
                df_static.loc[idx, c] = 91.4
            elif c == 'Gender':
                idx = x < 0
                df_static.loc[idx, c] = np.nan
            elif c == 'Height':
                idx = x < 0
                df_static.loc[idx, c] = np.nan

                # fix incorrectly recorded heights

                # 1.8 -> 180
                idx = x < 10
                df_static.loc[idx, c] = df_static.loc[idx, c] * 100

                # 18 -> 180
                idx = x < 25
                df_static.loc[idx, c] = df_static.loc[idx, c] * 10

                # 81.8 -> 180 (inch -> cm)
                idx = x < 100
                df_static.loc[idx, c] = df_static.loc[idx, c] * 2.2

                # 1800 -> 180
                idx = x > 1000
                df_static.loc[idx, c] = df_static.loc[idx, c] * 0.1

                # 400 -> 157
                idx = x > 250
                df_static.loc[idx, c] = df_static.loc[idx, c] * 0.3937

            elif c == 'Weight':
                idx = x < 35
                df_static.loc[idx, c] = np.nan

                idx = x > 299
                df_static.loc[idx, c] = np.nan


        df = delete_value(df, 'DiasABP', -1)
        df = replace_value(df, 'DiasABP', value=np.nan, below=1)
        df = replace_value(df, 'DiasABP', value=np.nan, above=200)
        df = replace_value(df, 'SysABP', value=np.nan, below=1)
        df = replace_value(df, 'MAP', value=np.nan, below=1)

        df = replace_value(df, 'NIDiasABP', value=np.nan, below=1)
        df = replace_value(df, 'NISysABP', value=np.nan, below=1)
        df = replace_value(df, 'NIMAP', value=np.nan, below=1)

        df = replace_value(df, 'HR', value=np.nan, below=1)
        df = replace_value(df, 'HR', value=np.nan, above=299)

        df = replace_value(df, 'PaCO2', value=np.nan, below=1)
        df = replace_value(df, 'PaCO2', value=lambda x: x * 10, below=10)

        df = replace_value(df, 'PaO2', value=np.nan, below=1)
        df = replace_value(df, 'PaO2', value=lambda x: x * 10, below=20)

        # the order of these steps matters
        df = replace_value(df, 'pH', value=lambda x: x * 10, below=0.8, above=0.65)
        df = replace_value(df, 'pH', value=lambda x: x * 0.1, below=80, above=65)
        df = replace_value(df, 'pH', value=lambda x: x * 0.01, below=800, above=650)
        df = replace_value(df, 'pH', value=np.nan, below=6.5)
        df = replace_value(df, 'pH', value=np.nan, above=8.0)

        # convert to farenheit
        df = replace_value(df, 'Temp', value=lambda x: x * 9 / 5 + 32, below=10, above=1)
        df = replace_value(df, 'Temp', value=lambda x: (x - 32) * 5 / 9, below=113, above=95)

        df = replace_value(df, 'Temp', value=np.nan, below=25)
        df = replace_value(df, 'Temp', value=np.nan, above=45)

        df = replace_value(df, 'RespRate', value=np.nan, below=1)
        df = replace_value(df, 'WBC', value=np.nan, below=1)

        df = replace_value(df, 'Weight', value=np.nan, below=35)
        df = replace_value(df, 'Weight', value=np.nan, above=299)


        df_full = pd.DataFrame(columns=['time', 'recordid']+feature_list+local_list)
        df_sampled = df.groupby(['recordid'])#, 'parameter'])
        for i, sample in enumerate(df_sampled):
            id = sample[0]
            df_signal = sample[1].groupby(['parameter'])
            signal_df = pd.DataFrame(columns=['time', 'recordid'])
            for j, signal_sample in enumerate(df_signal):
                param = signal_sample[0]
                sub_df = pd.DataFrame(columns=['time', 'recordid']+[param])
                sub_df[param] = signal_sample[1]['value']
                sub_df['recordid'] = id
                sub_df['time'] = signal_sample[1]['time']
                signal_df = signal_df.merge(sub_df, how='outer', on=['recordid', 'time'], sort=True, suffixes=[None, None])
            # Bin the values
            bins = pd.cut(signal_df.time, np.arange(signal_df['time'].iloc[0], signal_df['time'].iloc[-1], 60))
            col_list = list(signal_df.columns[2:])# - ['recordid', 'time']
            signal_df_binned =  pd.DataFrame(columns=signal_df.columns)
            signal_df_binned['time'] = np.arange(signal_df['time'].iloc[0], signal_df['time'].iloc[-1], 60)[:-1]
            signal_df_binned['recordid'] = id
            signal_df_binned[col_list] = signal_df.groupby(bins).agg(dict(zip(col_list, ["mean"]*len(col_list)))).to_numpy()#{"Temperature": "mean"})
            df_full = pd.concat([signal_df_binned, df_full])
        df_full.to_csv('./data/physionet/processed_df.csv')
        df_static.to_csv('./data/physionet/processed_static_df.csv')


    selected_features = ['DiasABP', 'GCS', 'HCT', 'MAP', 'NIDiasABP', 'NIMAP', 'NISysABP', 'RespRate', 'SysABP', 'Temp']

    # load in outcomes
    if dataset == 'set-a':
        y = pd.read_csv(os.path.join(data_dir, 'Outcomes-a.txt'))
    elif dataset == 'set-b':
        y = pd.read_csv(os.path.join(data_dir,  'Outcomes-.txt'))
    label_list = ['SAPS-I', 'SOFA', 'In-hospital_death']


    df_sampled = df_full.groupby(['recordid'])
    max_len = 80
    signals, signal_maps, signal_lens = [], [], []
    z_ls, z_gs = [], []
    for i, sample in enumerate(df_sampled):
        id = sample[0]
        x = sample[1][selected_features]
        if np.array(x.isna()).mean()>0.6 or len(x)<0.5*max_len:
            continue
        sample_map = x.isna().astype('float32')
        labels = y[y['RecordID']==id][label_list]
        z_l = sample[1][['MechVent']]
        x = x.fillna(0.0)
        z_g = df_static[df_static['RecordID']==id][['Age', 'Gender', 'Height', 'ICUType', 'Weight']]
        signals.append(np.array(x))
        signal_maps.append(np.array(sample_map))
        z_ls.append(np.array(z_l))
        z_gs.append(np.concatenate([np.array(z_g), np.array(labels)], axis=-1).reshape(-1,))
        signal_lens.append(min(max_len, len(x)))
    signals = tf.keras.preprocessing.sequence.pad_sequences(signals, maxlen=max_len, padding='post', value=0.0, dtype='float32')
    locals = tf.keras.preprocessing.sequence.pad_sequences(z_ls, maxlen=max_len, padding='post', value=0.0, dtype='float32')
    maps = tf.keras.preprocessing.sequence.pad_sequences(signal_maps, maxlen=max_len, padding='post', value=1.0, dtype='float32')
    z_gs = np.array(z_gs)
    signal_lens = np.array(signal_lens)

    test_inds = list(range(int(0.2*len(signals))))
    inds = list(range(int(0.2*len(signals)), len(signals)))
    rng = np.random.default_rng(seed)
    rng.shuffle(inds)
    train_inds = inds[:int(0.8*len(inds))]
    valid_inds = inds[int(0.8*len(inds)):]

    # plot a random sample
    ind = np.random.randint(0, len(train_inds))
    f, axs = plt.subplots(nrows=signals.shape[-1], ncols=1, figsize=(18, 14))
    for i, ax in enumerate(axs):
        ax.plot(signals[ind, :, i])
        ax.set_title(feature_list[i])
    plt.tight_layout()
    plt.savefig('./data/physionet/sample.png')

    x = signals[train_inds]
    print(x.shape)
    quit()

    train_signals, valid_signals, test_signals, normalization_specs = normalize_signals(signals, maps,
                                                                                        (train_inds, valid_inds, test_inds),
                                                                                        normalize)
    trainset = tf.data.Dataset.from_tensor_slices((train_signals, maps[train_inds], signal_lens[train_inds],
                                                   locals[train_inds], z_gs[train_inds])).batch(20)
    validset = tf.data.Dataset.from_tensor_slices((valid_signals, maps[valid_inds], signal_lens[valid_inds],
                                                   locals[valid_inds], z_gs[valid_inds])).batch(10)
    testset = tf.data.Dataset.from_tensor_slices((test_signals, maps[test_inds], signal_lens[test_inds],
                                                  locals[test_inds], z_gs[test_inds])).batch(30)
    return trainset, validset, testset, normalization_specs