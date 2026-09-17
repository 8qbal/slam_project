import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster

class VisualCoverageManager:
    """
    管理所有環境的柵格地圖 (Grid Map) 並計算覆蓋率獎勵。
    """
    def __init__(self, map_size=20.0, resolution=0.2, device="cuda"):
        self.map_size = map_size # 地圖邊長 (米)
        self.resolution = resolution # 格子大小 (米)
        self.grid_dim = int(map_size / resolution) # 格子數量 (例如 100x100)
        self.device = device
        
        # [num_envs, grid_dim, grid_dim]
        # 0 = 未知, 1 = 已知
        self.maps = None 
        self.initialized = False

    def init_maps(self, num_envs):
        if not self.initialized or self.maps.shape[0] != num_envs:
            self.maps = torch.zeros((num_envs, self.grid_dim, self.grid_dim), 
                                  dtype=torch.bool, device=self.device)
            self.initialized = True
            print(f"[CoverageManager] Initialized {num_envs} maps of size {self.grid_dim}x{self.grid_dim}")

    def reset_idx(self, env_ids):
        """重置指定環境的地圖 (當機器人死掉或重來時)"""
        if self.initialized and len(env_ids) > 0:
            self.maps[env_ids] = False

    def compute_reward(self, env, sensor_cfg: SceneEntityCfg):
        """
        核心邏輯：計算有多少射線打到了「未知區域」
        """
        # 1. 確保地圖已初始化
        if not self.initialized:
            self.init_maps(env.num_envs)
            
        # 2. 處理 Reset (檢查哪些環境剛被重置)
        # reset_buf 是 Isaac Lab 用來標記哪些環境剛重置的 buffer
        reset_env_ids = env.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        # 3. 取得 RayCaster 擊中點
        sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
        # ray_hits_w: [num_envs, num_rays, 3]
        hits = sensor.data.ray_hits_w.clone()
        
        # 4. 座標轉換 (World -> Grid)
        # 假設機器人出生點附近的區域是地圖範圍
        # 為了簡單，我們假設地圖中心跟隨機器人的出生點 (env_origins)
        # 或者是固定的 World Origin，這裡使用 World Origin 比較適合建圖任務
        
        # 將座標平移，使 (0,0) 位於地圖中心
        # grid_x = (x + size/2) / res
        half_size = self.map_size / 2.0
        
        # 我們只關心 X, Y 平面
        grid_indices = ((hits[..., :2] + half_size) / self.resolution).long()
        
        # 5. 過濾出界點
        x_idx = grid_indices[..., 0]
        y_idx = grid_indices[..., 1]
        valid_mask = (x_idx >= 0) & (x_idx < self.grid_dim) & \
                     (y_idx >= 0) & (y_idx < self.grid_dim)
        
        # 6. 計算獎勵 (向量化操作)
        # 我們要把 hits 展平成 [total_rays, 2] 來處理
        num_envs, num_rays = x_idx.shape
        
        # 為了利用 PyTorch 的高效率，我們使用 flatten 來索引
        batch_indices = torch.arange(num_envs, device=self.device).unsqueeze(1).expand(-1, num_rays)
        
        # 只取有效的點
        valid_batch = batch_indices[valid_mask]
        valid_x = x_idx[valid_mask]
        valid_y = y_idx[valid_mask]
        
        if len(valid_batch) == 0:
            return torch.zeros(num_envs, device=self.device)

        # 檢查這些點是否已經被訪問過
        # is_visited: [num_valid_hits]
        is_visited = self.maps[valid_batch, valid_x, valid_y]
        
        # 只有「未訪問 (False)」的點才給分
        new_visits = ~is_visited
        
        # 更新地圖
        # 注意：這裡可能會有同一個 step 多條射線打到同一個新格子的情況
        # 但為了效能，我們暫時允許重複計算 (或者你可以用 unique)
        self.maps[valid_batch[new_visits], valid_x[new_visits], valid_y[new_visits]] = True
        
        # 7. 聚合獎勵回每個環境
        # 我們需要計算每個 env 有多少個 new_visits
        rewards = torch.zeros(num_envs, device=self.device)
        
        # 使用 scatter_add 或 index_add 計算每個環境的新發現數
        # 此處 new_visits 是一個 mask，我們先找出這些新發現屬於哪個 env
        new_visit_envs = valid_batch[new_visits]
        
        # 每個新格子給 1.0 分 (之後在 config 裡調整 weight)
        ones = torch.ones_like(new_visit_envs, dtype=torch.float)
        rewards.scatter_add_(0, new_visit_envs, ones)
        
        # 正規化獎勵：除以射線總數，避免獎勵過大
        # 或是直接輸出數量，由 weight 控制
        return rewards

# 實例化一個全域管理器
COVERAGE_MANAGER = VisualCoverageManager(map_size=20.0, resolution=0.2)

# 定義給 Config 呼叫的 Wrapper 函數
def visual_coverage_reward(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    return COVERAGE_MANAGER.compute_reward(env, sensor_cfg)