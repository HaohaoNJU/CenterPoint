from ctypes import cdll
import numpy as np
import scipy.linalg
from copy import deepcopy
from threading import Lock


class UKFException(Exception):
    """Raise for errors in the UKF, usually due to bad inputs"""



def ctrv_process_model(x, dt, rand_var):
    """
    x : state_vector [  px. py, psi, v, dot_psi ]
    dt : delta time from last state timestamp, float by second 
    rand_var : [nu_a, nu_dot_psi], gaussions of acceleration and angle acceleration
   
     y
     ^
     |
     |
     |
    (.)-------------------> x
    z    Along z-axis anti-clockwise is the yaw rotation. 
    """
    assert len(x) == 5 and len(rand_var) == 2, "We need 5 dim state vectors and  2 randoms, nu_v and nu_psi_dot !"
    (px, py, psi,v, dpsi) = x  
    nu_a = rand_var[0] ; nu_ddpsi = rand_var[1] 
    tayler1 = np.zeros_like(x)
    tayler2 =  np.zeros_like(x)
    if np.abs(dpsi) > 0.001:
        tayler1[0] = v * (np.sin(psi + dpsi * dt)  - np.sin(psi)) / dpsi
        tayler1[1] = v * ( np.cos(psi) + np.cos(psi + dpsi * dt) ) / dpsi
        tayler1[2] = dpsi * dt 
    else:
    # if True:
        tayler1[0] = v * np.cos(psi) * dt
        tayler1[1] = v * np.sin(psi) * dt
        tayler1[2] = dpsi * dt 
    # pre-estimated terms , assuming dpsi=0, ddpsi=0, nu_ddpsi=0
    tayler2[0] = dt**2 * np.cos(psi) *  nu_a / 2
    tayler2[1] = dt**2 * np.sin(psi) * nu_a / 2
    tayler2[3] = nu_a * dt
    tayler2[2]  = dt**2 * nu_ddpsi / 2
    tayler2[4] = dt * nu_ddpsi 
    return  x + tayler1 + tayler2



class UKF:
    def __init__(self, num_states=5, num_aug = 2, initial_state = np.ones(5), initial_covar=np.eye(5), std_ddpsi = 1, std_a = 1, q_matrix = None, iterate_function = ctrv_process_model):
        """
        Initializes the unscented kalman filter
        :param num_states: int, the size of the state
        :param process_noise: the process noise covariance per unit time, should be num_states x num_states
        :param initial_state: initial values for the states, should be num_states x 1
        :param initial_covar: initial covariance matrix, should be num_states x num_states, typically large and diagonal
        :param k: UKF tuning parameter, typically 0 or 3 - num_states
        :param iterate_function: function that predicts the next state
                    takes in a num_states x 1 state and a float dt
                    returns a num_states x 1 state
        """
        assert num_states == len(initial_state), "Initial state should have the same num of elements as `num_states` !"
        self.n_dim = int(num_states)
        self.n_aug = self. n_dim + num_aug

        self.n_sig = 1 + self.n_aug * 2

        self.x = initial_state
        self.P = np.eye(self.n_dim) if initial_covar is None else initial_covar
        self.Q = np.zeros_like(self.P) if  q_matrix is None else q_matrix
        # params for measurement 
        self.K = None
        self.T = None
        self.S = None

        self.iterate = iterate_function

        self.lambd = 3 -  self.n_dim

        self.weights = np.zeros(self.n_sig)
        self.weights[0] = (self.lambd / (self.n_aug + self.lambd))
        self.weights[1:] = 1 / (2*(self.n_aug + self.lambd))

        # params for  random terms
        self.std_ddpsi = std_ddpsi
        self.std_a = std_a
        self.lock = Lock()

        self.sigmas = self.__get_sigmas()

    def __get_sigmas(self):
        """generates sigma points"""
        ret = np.zeros((self.n_aug, self.n_sig))
        P_aug = np.zeros((self.n_aug, self.n_aug),dtype=np.float64)
        P_aug[:self.n_dim, :self.n_dim] = self.P
        P_aug[self.n_dim, self.n_dim] = self.std_a * self.std_a
        P_aug[self.n_dim+1, self.n_dim+1] = self.std_ddpsi * self.std_ddpsi
        A = scipy.linalg.sqrtm(P_aug)

        x_aug = np.zeros(self.n_aug)
        x_aug[:self.n_dim] = self.x

        ret[:,0] = x_aug
        for i in range(self.n_aug):
            ret[:, i+1] = x_aug + np.sqrt(self.n_aug + self.lambd) * A[i]
            ret[:, i+1+self.n_aug] = x_aug - np.sqrt(self.n_aug + self.lambd)*A[i]


        return ret

    def update(self, state_idx, data, r_matrix=None):
        """
        performs a measurement update
        :param state_idx: list of indices (zero-indexed) of which state_idx were measured, that is, which are being updated
        :param data: list of the data corresponding to the values in state_idx
        :param r_matrix: error matrix for the data, again corresponding to the values in state_idx
        """

        
        assert len(state_idx) == len(data) , "state_idx idx nums shoule be the same with that of state value !"
        num_state_idx = len(state_idx)
        r_matrix = np.zeros((num_state_idx,num_state_idx)) if r_matrix is None else r_matrix
        assert r_matrix.shape== (num_state_idx, num_state_idx), 'Wrong r_matrix shape {}'.format(r_matrix.shape)

        
        # get predictions of measurements
        # y : d1 x (2d+1) ; self.sigmas : d x (2d+1)
        z_pred  = deepcopy(self.sigmas[state_idx] )
        # create y_mean, the mean of just the state_idx that are being updated
        # y_mean :  (d1,  ) 
        z_dct  = z_pred - self.x[state_idx].reshape(-1,1)
        x_dct = self.sigmas[:self.n_dim] - self.x.reshape(-1,1)
        # differences in y from y mean

        # covariance of measurement
        self.S = np.zeros((num_state_idx, num_state_idx))
        for ind in range(self.n_sig):
            val = z_dct[:, ind:ind+1]
            self.S +=  self.weights[ind] * val.dot(val.T)
         
        # add measurement noise
        # S matrix, measuarement covariance
        self.S += r_matrix

        # covariance of measurement with state_idx
        # T matrix, cross-correlation between sigma points in state space & measurement space 
        self.T = np.zeros((self.n_dim, num_state_idx))
        
        for ind in range(self.n_sig):
            val_state = x_dct[:,ind:ind+1]
            val_meas = z_dct[:, ind:ind+1].T
            self.T += self.weights[ind] * val_state.dot(val_meas)
        
        # Kalman Gain : (self.n_dim, num_state_idx)
        self.K = np.dot(self.T, np.linalg.inv(self.S))
        z_actual = data
        
        # posterior state
        self.x += np.dot(self.K, (z_actual -  self.x[state_idx]))
        # posterior covariance 
        # self.P -= np.dot(k, np.dot(p_yy, k.T))
        self.P -= np.dot(self.K, np.dot(self.S, self.K.T))

        

    def predict(self, dt):
        """
        performs a prediction step
        :param dt: float, amount of time since last prediction
        """

        
        # 1) generate the augmented sigma points
        self.sigmas = self.__get_sigmas()

        # 2) predict the sigma points 
        # sigmas_out : (self.n_dim, self.n_sig)
        sigmas_out = np.array([self.iterate(x[:self.n_dim], dt, x[self.n_dim : ]) for x in self.sigmas.T]).T

        # 3) predict mean and std  of state vector , (self.n_dim)
        x_out =  np.sum(sigmas_out * self.weights , axis = -1)
        
        p_out = np.zeros((self.n_dim, self.n_dim))
        dct_mat = sigmas_out - x_out.reshape(-1,1) # (self.n_dim, self.n_sig)
        for ind in range(self.n_sig):
            diff = dct_mat[:, ind].reshape(-1,1)
            p_out += self.weights[ind] * diff.dot(diff.T)
        # add process noise
        # p_out += dt * self.q # we choose to remove the process noise 
        
        # self.sigmas (self.n_aug, self.n_sig) ; sigmas_out (self.n_dim, self.n_sig)
        self.sigmas[:self.n_dim] = sigmas_out

        self.x = x_out

        self.P = p_out + self.Q
        




    def get_state(self, index=-1):
        """
        returns the current state (n_dim x 1), or a particular state variable (float)
        :param index: optional, if provided, the index of the returned variable
        :return:
        """
        if index >= 0:
            return self.x[index]
        elif isinstance(index,list) or isinstance(index,np.ndarray):
            return self.x[index]
        else:
            return self.x

    def get_covar(self):
        """
        :return: current state covariance (n_dim x n_dim)
        """
        return self.P

    def set_state(self, value, index=-1):
        """
        Overrides the filter by setting one variable of the state or the whole state
        :param value: the value to put into the state (1 x 1 or n_dim x 1)
        :param index: the index at which to override the state (-1 for whole state)
        """
        with self.lock:
            if index != -1:
                self.x[index] = value
            else:
                self.x = value

    def reset(self, state, covar):
        """
        Restarts the UKF at the given state and covariance
        :param state: n_dim x 1
        :param covar: n_dim x n_dim
        """
        with self.lock:
            self.x = state
            self.P = covar





            



            