import torch
import torch.nn as nn

from torchdiffeq import odeint_adjoint as odeint #已经install了

from .wrappers.cnf_regularization import RegularizedODEfunc

__all__ = ["CNF"]


class CNF(nn.Module): #reverse=F或默认，model(x)->z; reverse=T,model(z)->x
    def __init__(self, odefunc, T=1.0, train_T=False, regularization_fns=None, solver='dopri5', atol=1e-5, rtol=1e-5):
        super(CNF, self).__init__()
        if train_T:
            self.register_parameter("sqrt_end_time", nn.Parameter(torch.sqrt(torch.tensor(T))))
        else:
            self.register_buffer("sqrt_end_time", torch.sqrt(torch.tensor(T))) #不更新该参数但又可以保存
            #理解为模型的常数,在 buffers() 中的参数默认不会有梯度，parameters() 中的则相反;parameters()每次optim.step会得到更新，而不会更新buffers()
        #sqrt_end_time=torch.sqrt(torch.tensor(T))=tensor(1.)
        nreg = 0
        if regularization_fns is not None:
            odefunc = RegularizedODEfunc(odefunc, regularization_fns)
            nreg = len(regularization_fns)
        self.odefunc = odefunc
        self.nreg = nreg
        self.regularization_states = None
        self.solver = solver #dopri5
        self.atol = atol #1e-5
        self.rtol = rtol#1e-5
        self.test_solver = solver#dopri5
        self.test_atol = atol#1e-5
        self.test_rtol = rtol#1e-5
        self.solver_options = {}

    def forward(self, z, logpz=None, integration_times=None, reverse=False):
        #model(z, reverse=True) #是model(z)得到x
        #model(x, reverse=False)默认 #model(x)得到z

        if logpz is None:
            _logpz = torch.zeros(z.shape[0], 1).to(z) #logpz是0，z的行数行，1列
        else:
            _logpz = logpz

        if integration_times is None:
            integration_times = torch.tensor([0.0, self.sqrt_end_time * self.sqrt_end_time]).to(z)
            #tensor([0., 1.])
        if reverse:
            integration_times = _flip(integration_times, 0)
            #tensor([1., 0.])

        # Refresh the odefunc statistics.
        self.odefunc.before_odeint()

        # Add regularization states.
        reg_states = tuple(torch.tensor(0).to(z) for _ in range(self.nreg))

        if self.training:
            state_t = odeint( #得到adjoint的解
                self.odefunc,
                (z, _logpz) + reg_states,
                integration_times.to(z),
                atol=self.atol,
                rtol=self.rtol,
                method=self.solver,
                options=self.solver_options,
            ) #solving initial value problems (IVP),即得到z0
        else:
            state_t = odeint(
                self.odefunc,
                (z, _logpz),
                integration_times.to(z),
                atol=self.test_atol,
                rtol=self.test_rtol,
                method=self.test_solver,
            )

        if len(integration_times) == 2:
            state_t = tuple(s[1] for s in state_t)

        z_t, logpz_t = state_t[:2] 
        self.regularization_states = state_t[2:]

        if logpz is not None:
            return z_t, logpz_t
        else:
            return z_t #得到z_t

    def get_regularization_states(self):
        reg_states = self.regularization_states
        self.regularization_states = None
        return reg_states

    def num_evals(self):
        return self.odefunc._num_evals.item()


def _flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1, dtype=torch.long, device=x.device)
    return x[tuple(indices)]
