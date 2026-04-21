import numpy as np
import pyscipopt as scp
import torch
import os
import gurobipy as gp
from gurobipy import GRB
from collections import Counter
import random
import ecole


device=torch.device("cpu")

TASKS = ["IS", "IP", "WA", "MIS", "CA", "MVC", "SC", "MIPLIB"]
PROBLEM_CLASS = {"IS": 0, "IP": 1, "SC": 2, "MVC": 3, "CA": 4, "WA": 5, "BP": 6, "CFLP": 7, "GISP": 8, "MIS": 9, "LB": 10, "KS": 11}
GROUP_CLASS = {0: "IS", 1: "IP", 2: "SC", 3: "MVC", 4: "CA", 5: "WA", 6: "BP", 7: "CFLP",8: "GISP",9: "MIS", 10: "LB", 11: "KS"}
TASK_BATCH_SIZE = 1
ENERGY_WEIGHT_NORM = {"IP": 10, "WA": 100, "IS": -100, 'CA': -100000, 'SC': 100, "MVC": 100, "BP": -10000, "CFLP": 10000, "GISP": 1000, "MIS": -100, "LB": 100, "KS": -10000}
MAX_INIT_TASKS = {"MIS", "CA"}

def collect_test_instances(instance_dir, test_problem_type, difficulty):
    ins_dir = os.path.join(instance_dir, test_problem_type, difficulty)
    if not os.path.isdir(ins_dir):
        raise FileNotFoundError(f"Test instance directory does not exist: {ins_dir}")

    filenames = sorted(os.listdir(ins_dir))
    instances = [os.path.join(ins_dir, fn) for fn in filenames]
    print(f"Found {len(instances)} test instances (directory: {ins_dir})")
    return instances

def get_initial_best_obj(problem: str) -> float:
    return float("-inf") if problem in MAX_INIT_TASKS else float("inf")
def set_cpu_num(cpu_num):
    """
    Set the number of used cpu kernals.
    """
    os.environ['OMP_NUM_THREADS'] = str(cpu_num)
    os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    os.environ['MKL_NUM_THREADS'] = str(cpu_num)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    torch.set_num_threads(cpu_num)

def set_seed(seed):
    """
    Set the random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def get_group_stats(group_list, device):
    from collections import Counter
    counter = Counter(group_list)
    counts = torch.tensor(list(counter.values()), dtype=torch.float32, device=device)
    total = len(group_list)
    return {
        'n_groups': len(counter),
        'group_counts': counts,
        'group_frac': counts / total
    }
def position_get_ordered_flt(variable_features):
    lens=variable_features.shape[0]
    feature_widh=20 #max length 4095
    sorter=variable_features[:,1]
    position=torch.argsort(sorter)
    position=position/float(lens)
    
    position_feature=torch.zeros(lens,feature_widh)
    
    for row in range(position.shape[0]):
        flt_indx=position[row]
        divider=1.0
        for ks in range(feature_widh):
            if divider<=flt_indx:
                position_feature[row][ks]=1
                flt_indx-=divider
            divider/=2.0 
    position_feature=position_feature.to(device)
    variable_features=variable_features.to(device)
    v=torch.concat([variable_features,position_feature],dim=1)
    return v   

def get_BG_from_scip(ins_name):
    
    epsilon=1e-6
    
    #vars:  [obj coeff, norm_coeff, degree, max coeff, min coeff, Bin?]
    m=scp.Model()
    m.hideOutput(True)
    m.readProblem(ins_name)

    ncons=m.getNConss()
    nvars=m.getNVars()

    mvars=m.getVars()
    mvars.sort(key=lambda v:v.name)

    v_nodes=[]
    
    b_vars=[]
    
    ori_start=6
    emb_num=15
    
    for i in range(len(mvars)):
        tp=[0]*ori_start
        tp[3]=0
        tp[4]=1e+20
        #tp=[0,0,0,0,0]
        if mvars[i].vtype()=='BINARY':
            tp[ori_start-1]=1
            b_vars.append(i)
            
        v_nodes.append(tp)
    v_map={}
    
    for indx,v in enumerate(mvars):
        v_map[v.name]=indx

    obj=m.getObjective()
    obj_cons=[0]*(nvars+2)
    obj_node=[0,0,0,0]
    for e in obj:
        vnm=e.vartuple[0].name
        v=obj[e]
        v_indx=v_map[vnm]
        obj_cons[v_indx]=v
        v_nodes[v_indx][0]=v
        
        obj_node[0]+=v
        obj_node[1]+=1
    obj_node[0]/=obj_node[1]
    #quit()
       
    cons=m.getConss()
    new_cons=[]
    for cind,c in enumerate(cons):
        coeff=m.getValsLinear(c)
        if len(coeff)==0:
            #print(coeff,c)
            continue
        new_cons.append(c)
    cons=new_cons
    ncons=len(cons)
    cons_map=[[x,len(m.getValsLinear(x))] for x in cons]
    cons_map=sorted(cons_map,key=lambda x:[x[1],str(x[0])])
    cons=[x[0] for x in cons_map]

    A=[]
    for i in range(ncons):
        A.append([])
        for j in range(nvars+2):
            A[i].append(0)
    A.append(obj_cons)
    lcons=ncons
    c_nodes=[]
    for cind,c in enumerate(cons):
        coeff=m.getValsLinear(c)
        rhs=m.getRhs(c)
        lhs=m.getLhs(c)
        A[cind][-2]=rhs
        sense=0
        
        if rhs==lhs:
            sense=2
        elif rhs>=1e+20:
            sense=1
            rhs=lhs
        
        summation=0
        for k in coeff:
            v_indx=v_map[k]
            A[cind][v_indx]=1
            A[cind][-1]+=1
            v_nodes[v_indx][2]+=1
            v_nodes[v_indx][1]+=coeff[k]/lcons
            if v_indx==1066:
                print(coeff[k],lcons)
            v_nodes[v_indx][3]=max(v_nodes[v_indx][3],coeff[k])
            v_nodes[v_indx][4]=min(v_nodes[v_indx][4],coeff[k])
            #v_nodes[v_indx][3]+=cind*coeff[k]
            summation+=coeff[k]
        llc=max(len(coeff),1)
        c_nodes.append([summation/llc,llc,rhs,sense])
    c_nodes.append(obj_node)
    v_nodes=torch.as_tensor(v_nodes,dtype=torch.float32).to(device)    
    c_nodes=torch.as_tensor(c_nodes,dtype=torch.float32).to(device)
    b_vars=torch.as_tensor(b_vars,dtype=torch.int32).to(device)    
            
    A=np.array(A,dtype=np.float32)

    A=A[:,:-2]
    A=torch.as_tensor(A).to(device).to_sparse()
    clip_max=[20000,1,torch.max(v_nodes,0)[0][2].item()]
    clip_min=[0,-1,0]
    
    v_nodes[:,0]=torch.clamp(v_nodes[:,0],clip_min[0],clip_max[0])
    
    maxs=torch.max(v_nodes,0)[0]
    mins=torch.min(v_nodes,0)[0]
    diff=maxs-mins
    for ks in range(diff.shape[0]):
        if diff[ks]==0:
            diff[ks]=1
    v_nodes=v_nodes-mins
    v_nodes=v_nodes/diff
    v_nodes=torch.clamp(v_nodes,1e-5,1)
    v_nodes=position_get_ordered_flt(v_nodes)
    
    maxs=torch.max(c_nodes,0)[0]
    mins=torch.min(c_nodes,0)[0]
    diff=maxs-mins
    c_nodes=c_nodes-mins
    c_nodes=c_nodes/diff
    c_nodes=torch.clamp(c_nodes,1e-5,1)
    
    return A,v_map,v_nodes,c_nodes,b_vars       
    
    
def get_BG_from_GRB(ins_name):
    m=gp.read(ins_name)
    ori_start=6
    emb_num=15
    
    mvars=m.getVars()
    mvars.sort(key=lambda v:v.VarName)

    v_map={}
    for indx,v in enumerate(mvars):
        v_map[v.VarName]=indx

    nvars=len(mvars)
    
    v_nodes=[]
    b_vars=[]
    for i in range(len(mvars)):
        tp=[0]*ori_start
        tp[3]=0
        tp[4]=1e+20
        #tp=[0,0,0,0,0]
        if mvars[i].VType=='B':
            tp[ori_start-1]=1
            b_vars.append(i)
            
        v_nodes.append(tp)
    
    obj=m.getObjective()
    obj_cons=[0]*(nvars+2)
    obj_node=[0,0,0,0]
    
    nobjs=obj.size()
    for i in range(nobjs):
        vnm=obj.getVar(i).VarName
        v=obj.getCoeff(i)
        v_indx=v_map[vnm]
        obj_cons[v_indx]=v
        v_nodes[v_indx][0]=v
        obj_node[0]+=v
        obj_node[1]+=1
    obj_node[0]/=obj_node[1]
    
    cons=m.getConstrs()
    ncons=len(cons)
    lcons=ncons
    c_nodes=[]
    
    A=[]
    for i in range(ncons):
        A.append([])
        for j in range(nvars+2):
            A[i].append(0)
    A.append(obj_cons)
    for i in range(ncons):
        tmp_v=[]
        tmp_c=[]
        
        sense=cons[i].Sense
        rhs=cons[i].RHS
        nzs=0
        
        if sense=='<':
            sense=0
        elif sense=='>':
            sense=1
        elif sense=='=':
            sense=2
            
        tmp_c=[0,0,rhs,sense]
        summation=0
        tmp_v=[0,0,0,0,0]
        for v in mvars:
            v_indx=v_map[v.VarName]
            ce=m.getCoeff(cons[i],v)
            
            if ce!=0:
                nzs+=1
                summation+=ce
                A[i][v_indx]=1
                A[i][-1]+=1
            
        if nzs==0:
            continue
        tmp_c[0]=summation/nzs
        tmp_c[1]=nzs
        c_nodes.append(tmp_c)
        for v in mvars:
            v_indx=v_map[v.VarName]
            ce=m.getCoeff(cons[i],v)
            
            if ce!=0:            
                v_nodes[v_indx][2]+=1
                v_nodes[v_indx][1]+=ce/lcons
                v_nodes[v_indx][3]=max(v_nodes[v_indx][3],ce)
                v_nodes[v_indx][4]=min(v_nodes[v_indx][4],ce)
    
    c_nodes.append(obj_node)
    v_nodes=torch.as_tensor(v_nodes,dtype=torch.float32).to(device)    
    c_nodes=torch.as_tensor(c_nodes,dtype=torch.float32).to(device)
    b_vars=torch.as_tensor(b_vars,dtype=torch.int32).to(device)    

    A=np.array(A,dtype=np.float32)

    A=A[:,:-2]
    A=torch.as_tensor(A).to(device).to_sparse()
    clip_max=[20000,1,torch.max(v_nodes,0)[0][2].item()]
    clip_min=[0,-1,0]
    
    v_nodes[:,0]=torch.clamp(v_nodes[:,0],clip_min[0],clip_max[0])
    
    maxs=torch.max(v_nodes,0)[0]
    mins=torch.min(v_nodes,0)[0]
    diff=maxs-mins
    for ks in range(diff.shape[0]):
        if diff[ks]==0:
            diff[ks]=1
    v_nodes=v_nodes-mins
    v_nodes=v_nodes/diff
    v_nodes=torch.clamp(v_nodes,1e-5,1)
    #v_nodes=position_get_ordered(v_nodes)
    v_nodes=position_get_ordered_flt(v_nodes)
    
    maxs=torch.max(c_nodes,0)[0]
    mins=torch.min(c_nodes,0)[0]
    diff=maxs-mins
    c_nodes=c_nodes-mins
    c_nodes=c_nodes/diff
    c_nodes=torch.clamp(c_nodes,1e-5,1)
    
    return A,v_map,v_nodes,c_nodes,b_vars       
    
def get_a_new2(ins_name):
    epsilon = 1e-6

    m = scp.Model()
    m.hideOutput(True)
    m.readProblem(ins_name)

    ncons = m.getNConss()
    nvars = m.getNVars()

    mvars = m.getVars()
    mvars.sort(key=lambda v: v.name)

    v_nodes = []

    b_vars = []

    ori_start = 6
    emb_num = 15

    for i in range(len(mvars)):
        tp = [0] * ori_start
        tp[3] = 0
        tp[4] = 1e+20
        # tp=[0,0,0,0,0]
        if mvars[i].vtype() == 'BINARY':
            tp[ori_start - 1] = 1
            b_vars.append(i)

        v_nodes.append(tp)
    v_map = {}

    for indx, v in enumerate(mvars):
        v_map[v.name] = indx

    obj = m.getObjective()
    obj_cons = [0] * (nvars + 2)
    indices_spr = [[], []]
    values_spr = []
    obj_node = [0, 0, 0, 0]
    for e in obj:
        vnm = e.vartuple[0].name
        v = obj[e]
        v_indx = v_map[vnm]
        obj_cons[v_indx] = v
        if v != 0:
            indices_spr[0].append(0)
            indices_spr[1].append(v_indx)
            # values_spr.append(v)
            values_spr.append(1)
        v_nodes[v_indx][0] = v

        obj_node[0] += v
        obj_node[1] += 1
    obj_node[0] /= obj_node[1]

    cons = m.getConss()
    new_cons = []

    for cind, c in enumerate(cons):
        coeff = m.getValsLinear(c)
        if len(coeff) == 0:
            # print(coeff,c)
            continue
        new_cons.append(c)
    cons = new_cons
    ncons = len(cons)
    cons_map = [[x, len(m.getValsLinear(x))] for x in cons]

    cons_map = sorted(cons_map, key=lambda x: [x[1], str(x[0])])
    cons = [x[0] for x in cons_map]

    lcons = ncons
    c_nodes = []
    for cind, c in enumerate(cons):
        coeff = m.getValsLinear(c)
        rhs = m.getRhs(c)
        lhs = m.getLhs(c)
        sense = 0

        if rhs == lhs:
            sense = 2
        elif rhs >= 1e+20:
            sense = 1
            rhs = lhs

        summation = 0
        for k in coeff:
            v_indx = v_map[k]
            if coeff[k] != 0:
                indices_spr[0].append(cind)
                indices_spr[1].append(v_indx)
                values_spr.append(1)
            v_nodes[v_indx][2] += 1
            v_nodes[v_indx][1] += coeff[k] / lcons
            v_nodes[v_indx][3] = max(v_nodes[v_indx][3], coeff[k])
            v_nodes[v_indx][4] = min(v_nodes[v_indx][4], coeff[k])
            summation += coeff[k]
        llc = max(len(coeff), 1)
        c_nodes.append([summation / llc, llc, rhs, sense])
    c_nodes.append(obj_node)
    v_nodes = torch.as_tensor(v_nodes, dtype=torch.float32).to(device)
    c_nodes = torch.as_tensor(c_nodes, dtype=torch.float32).to(device)
    b_vars = torch.as_tensor(b_vars, dtype=torch.int32).to(device)

    A = torch.sparse_coo_tensor(indices_spr, values_spr, (ncons + 1, nvars)).to(device)
    clip_max = [20000, 1, torch.max(v_nodes, 0)[0][2].item()]
    clip_min = [0, -1, 0]

    v_nodes[:, 0] = torch.clamp(v_nodes[:, 0], clip_min[0], clip_max[0])

    maxs = torch.max(v_nodes, 0)[0]
    mins = torch.min(v_nodes, 0)[0]
    diff = maxs - mins
    for ks in range(diff.shape[0]):
        if diff[ks] == 0:
            diff[ks] = 1
    v_nodes = v_nodes - mins
    v_nodes = v_nodes / diff
    v_nodes = torch.clamp(v_nodes, 1e-5, 1)

    maxs = torch.max(c_nodes, 0)[0]
    mins = torch.min(c_nodes, 0)[0]
    diff = maxs - mins
    c_nodes = c_nodes - mins
    c_nodes = c_nodes / diff
    c_nodes = torch.clamp(c_nodes, 1e-5, 1)

    return A, v_map, v_nodes, c_nodes, b_vars

def get_diffilo_data_train(ins_name):
    # extract the features from the instance file
    m = scp.Model()
    m.hideOutput(True)
    m.readProblem(ins_name)

    ncons = m.getNConss()
    nvars = m.getNVars()
    mvars = m.getVars()

    variable_features = []

    for i in range(len(mvars)):
        tp = [0] * 5
        tp[3] = 0
        tp[4] = 1e+20

        variable_features.append(tp)
        
    v_map = {}
    for indx, v in enumerate(mvars):
        v_map[v.name] = indx

    obj = m.getObjective()
    obj_cons = [0] * (nvars + 2)
    indices_spr = [[], []]
    values_spr = []
    obj_node = [0, 0, 0]
    for e in obj:
        vnm = e.vartuple[0].name
        v = obj[e]
        v_indx = v_map[vnm]
        obj_cons[v_indx] = v
        if v != 0:
            indices_spr[0].append(0)
            indices_spr[1].append(v_indx)
            values_spr.append(1)
        variable_features[v_indx][0] = v
        obj_node[0] += v
        obj_node[1] += 1
        
    if obj_node[1] > 0:
        obj_node[0] /= obj_node[1]

    cons = m.getConss()
    new_cons = []
    for cind, c in enumerate(cons):
        coeff = m.getValsLinear(c)
        if len(coeff) == 0:
            continue
        new_cons.append(c)
    cons = new_cons
    ncons = len(cons)

    lcons = ncons
    constraint_features = []
    for cind, c in enumerate(cons):
        coeff = m.getValsLinear(c)
        rhs = m.getRhs(c)
        lhs = m.getLhs(c)

        summation = 0
        for k in coeff:
            v_indx = v_map[k]
            if coeff[k] != 0:
                indices_spr[0].append(cind)
                indices_spr[1].append(v_indx)
                values_spr.append(1)
            variable_features[v_indx][2] += 1
            variable_features[v_indx][1] += coeff[k] / lcons
            variable_features[v_indx][3] = max(variable_features[v_indx][3], coeff[k])
            variable_features[v_indx][4] = min(variable_features[v_indx][4], coeff[k])
            summation += coeff[k]
        llc = max(len(coeff), 1)
        constraint_features.append([summation / llc, llc, rhs])
    
    variable_features = torch.as_tensor(variable_features, dtype=torch.float32)
    constraint_features = torch.as_tensor(constraint_features, dtype=torch.float32)

    A = torch.sparse_coo_tensor(indices_spr, values_spr, (ncons, nvars))
    clip_max = [20000, 1, torch.max(variable_features, 0)[0][2].item()]
    clip_min = [0, -1, 0]

    variable_features[:, 0] = torch.clamp(variable_features[:, 0], clip_min[0], clip_max[0])

    maxs = torch.max(variable_features, 0)[0]
    mins = torch.min(variable_features, 0)[0]
    diff = maxs - mins
    for ks in range(diff.shape[0]):
        if diff[ks] == 0:
            diff[ks] = 1
    variable_features = variable_features - mins
    variable_features = variable_features / diff

    maxs = torch.max(constraint_features, 0)[0]
    mins = torch.min(constraint_features, 0)[0]
    diff = maxs - mins
    for ks in range(diff.shape[0]):
        if diff[ks] == 0:
            diff[ks] = 1
    constraint_features = constraint_features - mins
    constraint_features = constraint_features / diff

    # save the preprocessed data
    edge_indices= torch.LongTensor(np.array(indices_spr, dtype=int))
    edge_features = torch.FloatTensor(values_spr).reshape(-1,1)
    graph = [constraint_features, edge_indices, edge_features, variable_features]

    # save the tensor data
    # A, b, c in the standard form min c^T x s.t. Ax <= b 
    model = ecole.scip.Model.from_file(ins_name)
    obs = ecole.observation.MilpBipartite().extract(model, True)
    A_i = torch.LongTensor(np.array(obs.edge_features.indices, dtype=int))
    A_e = torch.FloatTensor(obs.edge_features.values)
    A = torch.sparse_coo_tensor(A_i, A_e).to_dense()         
    c = torch.FloatTensor(obs.variable_features[:,0].reshape(-1,1))
    b = torch.FloatTensor(obs.constraint_features)

    return graph, A, b, c

def get_diffilo_data_test(ins_name):
    from src.dataloader.graph_dataset import BipartiteNodeData
    m = scp.Model()
    m.hideOutput(True)
    m.readProblem(ins_name)

    ncons = m.getNConss()
    nvars = m.getNVars()

    mvars = m.getVars()

    variable_features = []

    for i in range(len(mvars)):
        tp = [0] * 5
        tp[3] = 0
        tp[4] = 1e+20

        variable_features.append(tp)
    v_map = {}

    for indx, v in enumerate(mvars):
        v_map[v.name] = indx

    obj = m.getObjective()
    obj_cons = [0] * (nvars + 2)
    indices_spr = [[], []]
    values_spr = []
    obj_node = [0, 0, 0]
    for e in obj:
        vnm = e.vartuple[0].name
        v = obj[e]
        v_indx = v_map[vnm]
        obj_cons[v_indx] = v
        if v != 0:
            indices_spr[0].append(0)
            indices_spr[1].append(v_indx)
            values_spr.append(1)
        variable_features[v_indx][0] = v

        obj_node[0] += v
        obj_node[1] += 1
    obj_node[0] /= obj_node[1]

    cons = m.getConss()
    new_cons = []
    for cind, c in enumerate(cons):
        coeff = m.getValsLinear(c)
        if len(coeff) == 0:
            continue
        new_cons.append(c)
    cons = new_cons
    ncons = len(cons)

    lcons = ncons
    constraint_features = []
    for cind, c in enumerate(cons):
        coeff = m.getValsLinear(c)
        rhs = m.getRhs(c)
        lhs = m.getLhs(c)
        
        summation = 0
        for k in coeff:
            v_indx = v_map[k]
            if coeff[k] != 0:
                indices_spr[0].append(cind)
                indices_spr[1].append(v_indx)
                values_spr.append(1)
            variable_features[v_indx][2] += 1
            variable_features[v_indx][1] += coeff[k] / lcons
            variable_features[v_indx][3] = max(variable_features[v_indx][3], coeff[k])
            variable_features[v_indx][4] = min(variable_features[v_indx][4], coeff[k])
            summation += coeff[k]
        llc = max(len(coeff), 1)
        constraint_features.append([summation / llc, llc, rhs])
    variable_features = torch.as_tensor(variable_features, dtype=torch.float32)
    constraint_features = torch.as_tensor(constraint_features, dtype=torch.float32)

    A = torch.sparse_coo_tensor(indices_spr, values_spr, (ncons, nvars))
    clip_max = [20000, 1, torch.max(variable_features, 0)[0][2].item()]
    clip_min = [0, -1, 0]

    variable_features[:, 0] = torch.clamp(variable_features[:, 0], clip_min[0], clip_max[0])

    maxs = torch.max(variable_features, 0)[0]
    mins = torch.min(variable_features, 0)[0]
    diff = maxs - mins
    for ks in range(diff.shape[0]):
        if diff[ks] == 0:
            diff[ks] = 1
    variable_features = variable_features - mins
    variable_features = variable_features / diff

    maxs = torch.max(constraint_features, 0)[0]
    mins = torch.min(constraint_features, 0)[0]
    diff = maxs - mins
    for ks in range(diff.shape[0]):
        if diff[ks] == 0:
            diff[ks] = 1
    constraint_features = constraint_features - mins
    constraint_features = constraint_features / diff
    
    edge_indices= torch.LongTensor(np.array(indices_spr, dtype=int))
    edge_features = torch.FloatTensor(values_spr).reshape(-1,1)
    
    graph = BipartiteNodeData(constraint_features, edge_indices, edge_features, variable_features).cuda()
    
    m = ecole.scip.Model.from_file(ins_name)
    obs = ecole.observation.MilpBipartite().extract(m, True)

    constraint_features = torch.FloatTensor(obs.constraint_features)
    edge_indices = torch.LongTensor(np.array(obs.edge_features.indices, dtype=int))
    edge_features = torch.FloatTensor(obs.edge_features.values.reshape((-1,1)))
    variable_features = torch.FloatTensor(obs.variable_features)

    b = torch.FloatTensor(obs.constraint_features).numpy()
    A_i = torch.LongTensor(np.array(obs.edge_features.indices, dtype=int))
    A_e = torch.FloatTensor(obs.edge_features.values.reshape(-1))
    c = torch.FloatTensor(obs.variable_features[:,0].reshape(-1,1)).numpy()
    A = torch.sparse_coo_tensor(A_i, A_e).to_dense().numpy()

    return graph, A, b, c