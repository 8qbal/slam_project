import torch
from isaaclab.managers import SceneEntityCfg

# 這是一個全域變數，用來暫存每個環境的"訪問紀錄"
# 格式: [env_id, x_grid, y_grid]
# 為了效能，我們簡化為：紀錄機器人是否移動到了新的座標區塊
VISITED_MAPS = None

def reset_exploration_map(env_ids: torch.Tensor, num_envs: int, device: str):
    """重置環境時，清空該環境的探索地圖"""
    global VISITED_MAPS
    # 假設地圖範圍是 20m x 20m，解析度 0.5m -> 40x40 的 grid
    grid_size = 40 
    
    if VISITED_MAPS is None or VISITED_MAPS.shape[0] != num_envs:
        # 初始化所有環境的地圖 (全是 0)
        VISITED_MAPS = torch.zeros((num_envs, grid_size, grid_size), device=device, dtype=torch.bool)
    
    # 清空被 Reset 的環境
    if len(env_ids) > 0:
        VISITED_MAPS[env_ids] = False

def exploration_reward(env, grid_resolution: float = 0.5, map_range: float = 20.0) -> torch.Tensor:
    """
    探索獎勵：如果機器人進入了一個新的格子，給分！
    """
    global VISITED_MAPS
    
    # 1. 初始化檢查
    if VISITED_MAPS is None:
        reset_exploration_map([], env.num_envs, env.device)
        
    # 2. 取得機器人位置 (x, y)
    # root_pos_w: [num_envs, 3]
    pos = env.scene["robot"].data.root_pos_w[:, :2] 
    
    # 3. 轉換為 Grid 座標 (將 -10m~10m 映射到 0~40 格)
    # map_range/2 是偏移量，讓 (0,0) 在地圖中心
    grid_indices = ((pos + (map_range / 2.0)) / grid_resolution).long()
    
    # 4. 邊界檢查 (防止出界 crash)
    grid_size = VISITED_MAPS.shape[1]
    valid_mask = (grid_indices[:, 0] >= 0) & (grid_indices[:, 0] < grid_size) & \
                 (grid_indices[:, 1] >= 0) & (grid_indices[:, 1] < grid_size)
    
    # 5. 計算獎勵
    rewards = torch.zeros(env.num_envs, device=env.device)
    
    # 只處理在邊界內的環境
    valid_indices = torch.nonzero(valid_mask).squeeze()
    if len(valid_indices) > 0:
        # 取得這些環境的 x, y 索引
        x_idx = grid_indices[valid_indices, 0]
        y_idx = grid_indices[valid_indices, 1]
        
        # 檢查是否已經訪問過
        # 注意：這裡利用 PyTorch 的高級索引功能
        is_visited = VISITED_MAPS[valid_indices, x_idx, y_idx]
        
        # 如果沒訪問過 (False)，就給獎勵，並標記為 True
        new_visit_mask = ~is_visited
        
        # [關鍵] 給予獎勵！
        # 發現新區塊 = +1.0 分
        rewards[valid_indices[new_visit_mask]] = 1.0 
        
        # 更新地圖
        VISITED_MAPS[valid_indices[new_visit_mask], x_idx[new_visit_mask], y_idx[new_visit_mask]] = True
        
    return rewards