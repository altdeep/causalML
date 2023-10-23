from copy import deepcopy
from pgmpy.factors.discrete.CPD import TabularCPD


ENDOGENOUS_VARS = ["Host Door Selection", "Strategy", "2nd Choice", "Win or Lose"]

def clone(cpd: TabularCPD, suffix: str="Hyp"):
    suffix = " " + suffix
    endogenous_vars = ENDOGENOUS_VARS
    cpd_hyp = deepcopy(cpd)
    cpd_hyp.variable += suffix
    variables = cpd_hyp.variables
    new_variables = []
    new_state_names = {}
    new_name_to_no = {}
    new_no_to_name = {}
    for var_name, val in cpd_hyp.state_names.items():
        if var_name in endogenous_vars:
            new_variables.append(var_name + suffix)
            new_state_names[var_name + " Hyp"] = val
        else:
            new_variables.append(var_name)
            new_state_names[var_name] = val
    for var_name, val in cpd_hyp.name_to_no.items():
        if var_name in endogenous_vars:
            new_name_to_no[var_name + suffix] = val
        else:
            new_name_to_no[var_name] = val
    for var_name, val in cpd_hyp.no_to_name.items():
        if var_name in endogenous_vars:
            new_no_to_name[var_name + suffix] = val
        else:
            new_no_to_name[var_name] = val
    cpd_hyp.variables = new_variables
    cpd_hyp.state_names = new_state_names
    cpd_hyp.name_to_no = new_name_to_no
    cpd_hyp.no_to_name = new_no_to_name
    return cpd_hyp
