import os
import random
import torch_geometric
from typing import Optional, Union, List, Tuple
from .graph_dataset import GraphDataset
from src.utils.utils import PROBLEM_CLASS

def create_dataloaders(
    method_type: str,
    problem_type: Union[str, List[str]],
    data_dir: str = "./dataset",
    difficulty = "hard", 
    solve_time = "3600s",
    batch_size: int = 1,
    num_workers: int = 0,
    train_split: float = 0.7,
    solver_settings: str = "gurobi",
    train_shuffle: bool = True,
    val_shuffle: bool = False,
    problem_class_map: Optional[dict] = None,
    use_edge_coeff: bool = True,
) -> Tuple[torch_geometric.loader.DataLoader, torch_geometric.loader.DataLoader]:
    """
    Create data loaders
    
    Args:
        method_type: Method type, can be 'ps', 'coco', 'diffilo', or 'apollo'
        problem_type: Problem type, can be a single string (e.g., 'IS') or a list (e.g., ['IS', 'IP', 'SC'])
        data_dir: Root directory of the dataset
        difficulty: Difficulty level of the dataset
        solve_time: Time used by the professional solver to solve the training set
        batch_size: Batch size
        num_workers: Number of data loading worker threads
        train_split: Proportion of training set
        solver_settings: Solver configuration
        train_shuffle: Whether to shuffle the training set
        val_shuffle: Whether to shuffle the validation set
        problem_class_map: Mapping from problem type to group ID
        
    Returns:
        (train_loader, val_loader): Training loader, validation loader
    """
    if problem_class_map is None:
        problem_class_map = PROBLEM_CLASS
    
    # Support single problem type or multiple problem types
    if isinstance(problem_type, str):
        problem_types = [problem_type]
    else:
        problem_types = problem_type
    
    sample_files = []
    if method_type == 'DiffILO':
        dir_bg = os.path.join(data_dir, solver_settings, problem_type, difficulty, solve_time, 'samples')

        if not os.path.exists(dir_bg) or not os.path.exists(dir_bg):
            print(f"Warning: skipping {pt}, directory not found")
            
        sample_files = [os.path.join(dir_bg, f) for f in os.listdir(dir_bg) if f.endswith('.pkl')]
    else:
        for pt in problem_types:
            pt_group = problem_class_map.get(pt, 20)
        
            dir_bg = os.path.join(data_dir, pt, difficulty, solve_time, 'BG')
            dir_sol = os.path.join(data_dir, pt, difficulty, solve_time, 'solutions')
        
            if not os.path.exists(dir_bg) or not os.path.exists(dir_sol):
                print(f"Warning: skipping {pt}, directory not found")
                continue
            
            bg_files = [f for f in os.listdir(dir_bg) if f.endswith('.bg')]
            
            for name in bg_files:
                bg_file = os.path.join(dir_bg, name)
                sol_file = os.path.join(dir_sol, name.replace('.bg', '.sol'))
                
                if os.path.exists(sol_file):
                    sample_files.append((bg_file, sol_file, pt_group))
        
    random.shuffle(sample_files)
    split_idx = int(train_split * len(sample_files))
    train_files = sample_files[:split_idx]
    val_files = sample_files[split_idx:]
    
    train_dataset = GraphDataset(train_files, method_type, use_edge_coeff=use_edge_coeff)
    val_dataset = GraphDataset(val_files, method_type, use_edge_coeff=use_edge_coeff)
    
    if method_type == 'DiffILO':
        train_loader = torch_geometric.loader.DataLoader(
            train_dataset,
            shuffle=train_shuffle,
            batch_size=batch_size,
            follow_batch=["constraint_features", "variable_features"],
            num_workers=1
        )

        val_loader = torch_geometric.loader.DataLoader(
            val_dataset,
            shuffle=val_shuffle, 
            batch_size=batch_size,
            follow_batch=["constraint_features", "variable_features"],
            num_workers=1
        )
    else:
        train_loader = torch_geometric.loader.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=train_shuffle,
            num_workers=num_workers
        )
        
        val_loader = torch_geometric.loader.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=val_shuffle,
            num_workers=num_workers
        )
    
    return train_loader, val_loader