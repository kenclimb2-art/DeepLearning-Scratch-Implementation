import cv2
import numpy as np
import os
import pickle

# ==========================================
# 1. ユーティリティ (im2col / col2im)
# ==========================================
def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_data.shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col

def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)
    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]
    return img[:, :, pad:H + pad, pad:W + pad]

# ==========================================
# 2. レイヤ定義
# ==========================================
class BatchNormalization:
    def __init__(self, gamma, beta, momentum=0.9, running_mean=None, running_var=None):
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.input_shape = None 
        self.running_mean = running_mean
        self.running_var = running_var  
        self.batch_size = None
        self.xc = None
        self.std = None
        self.dgamma = None
        self.dbeta = None

    def forward(self, x, train_flg=True):
        self.input_shape = x.shape
        if x.ndim == 4:
            N, C, H, W = x.shape
            x = x.transpose(0, 2, 3, 1) # (N, H, W, C)
            x = x.reshape(-1, C)        # (N*H*W, C)
        out = self.__forward(x, train_flg)
        if len(self.input_shape) == 4:
            N, C, H, W = self.input_shape
            out = out.reshape(N, H, W, C)
            out = out.transpose(0, 3, 1, 2) 
        return out

    def __forward(self, x, train_flg):
        if self.running_mean is None:
            N, D = x.shape
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D)
        if train_flg:
            mu = x.mean(axis=0)
            xc = x - mu
            var = np.mean(xc**2, axis=0)
            std = np.sqrt(var + 10e-7)
            xn = xc / std
            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std
            self.running_mean = self.momentum * self.running_mean + (1-self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1-self.momentum) * var            
        else:
            xc = x - self.running_mean
            xn = xc / ((np.sqrt(self.running_var + 10e-7)))
        out = self.gamma * xn + self.beta 
        return out

    def backward(self, dout):
        if dout.ndim == 4:
            N, C, H, W = dout.shape
            dout = dout.transpose(0, 2, 3, 1)
            dout = dout.reshape(-1, C)
        dx = self.__backward(dout)
        if self.input_shape is not None and len(self.input_shape) == 4:
            N, C, H, W = self.input_shape
            dx = dx.reshape(N, H, W, C)
            dx = dx.transpose(0, 3, 1, 2)
        return dx

    def __backward(self, dout):
        dbeta = dout.sum(axis=0)
        dgamma = np.sum(self.xn * dout, axis=0)
        dxn = self.gamma * dout
        dxc = dxn / self.std
        dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        dmu = np.sum(dxc, axis=0)
        dx = dxc - dmu / self.batch_size
        self.dgamma = dgamma
        self.dbeta = dbeta
        return dx

class Relu:
    def __init__(self):
        self.mask = None
    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out
    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx

class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.original_x_shape = None
        self.dW = None
        self.db = None
    def forward(self, x):
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x
        out = np.dot(self.x, self.W) + self.b
        return out
    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        dx = dx.reshape(*self.original_x_shape)
        return dx

class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None
    def forward(self, x, t):
        self.t = t
        self.y = self._softmax(x)
        self.loss = self._cross_entropy_error(self.y, self.t)
        return self.loss
    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size: 
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size
        return dx
    def _softmax(self, x):
        if x.ndim == 2:
            x = x.T
            x = x - np.max(x, axis=0)
            y = np.exp(x) / np.sum(np.exp(x), axis=0)
            return y.T
        x = x - np.max(x)
        return np.exp(x) / np.sum(np.exp(x))
    def _cross_entropy_error(self, y, t):
        if y.ndim == 1:
            t = t.reshape(1, t.size)
            y = y.reshape(1, y.size)
        if t.size == y.size:
            t = t.argmax(axis=1)
        batch_size = y.shape[0]
        return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad
        self.x = None   
        self.col = None
        self.col_W = None
        self.dW = None
        self.db = None
    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = 1 + int((H + 2*self.pad - FH) / self.stride)
        out_w = 1 + int((W + 2*self.pad - FW) / self.stride)
        col = im2col(x, FH, FW, self.stride, self.pad)
        col_W = self.W.reshape(FN, -1).T
        out = np.dot(col, col_W) + self.b
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)
        self.x = x
        self.col = col
        self.col_W = col_W
        return out
    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0,2,3,1).reshape(-1, FN)
        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)
        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)
        return dx

class Pooling:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad
        self.x = None
        self.arg_max = None
    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)
        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h*self.pool_w)
        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)
        self.x = x
        self.arg_max = arg_max
        return out
    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)
        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,)) 
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)
        return dx

# ==========================================
# 3. Three_ConvNet (3層畳み込み + BatchNorm)
# ==========================================
# 5年前の成功モデルの構成を再現
# Conv(32) -> BN -> ReLU -> Pool
# Conv(64) -> BN -> ReLU -> Pool
# Conv(64) -> BN -> ReLU -> Pool
# Affine(100) -> BN -> ReLU
# Affine(15)
class Three_ConvNet:
    def __init__(self, input_dim=(1, 28, 28), 
                 conv_param={'filter_size':3, 'pad':1, 'stride':1},
                 pool_param={'pool_size':2, 'pad':0, 'stride':2},
                 hidden_size=100, output_size=15):
        
        self.params = {}
        # フィルタ数リスト [32, 64, 64]
        
        # --- 重みの初期化 (Heの初期値) ---
        # Conv1: 1ch -> 32ch
        pre_node_1 = 1 * 3 * 3
        self.params['W1'] = np.random.randn(32, 1, 3, 3) * np.sqrt(2.0/pre_node_1)
        self.params['b1'] = np.zeros(32)
        self.params['gamma1'] = np.ones(32)
        self.params['beta1'] = np.zeros(32)

        # Conv2: 32ch -> 64ch
        pre_node_2 = 32 * 3 * 3
        self.params['W2'] = np.random.randn(64, 32, 3, 3) * np.sqrt(2.0/pre_node_2)
        self.params['b2'] = np.zeros(64)
        self.params['gamma2'] = np.ones(64)
        self.params['beta2'] = np.zeros(64)

        # Conv3: 64ch -> 64ch
        pre_node_3 = 64 * 3 * 3
        self.params['W3'] = np.random.randn(64, 64, 3, 3) * np.sqrt(2.0/pre_node_3)
        self.params['b3'] = np.zeros(64)
        self.params['gamma3'] = np.ones(64)
        self.params['beta3'] = np.zeros(64)

        # 全結合層
        # プーリング後のサイズ計算:
        # 28x28 ->(Pool)-> 14x14 ->(Pool)-> 7x7 ->(Pool)-> 3x3 (約)
        # 正確には: 
        # L1: 28->28(conv)->14(pool)
        # L2: 14->14(conv)->7(pool)
        # L3: 7 -> 7(conv)->3(pool) (padding等の関係で3か4になるが、計算上3x3になる設定)
        # Poolingを1回サボるので、7x7 (49ピクセル) のまま入ってくる
        pool_out_size = 64 * 7 * 7
        
        self.params['W4'] = np.random.randn(pool_out_size, hidden_size) * np.sqrt(2.0/pool_out_size)
        self.params['b4'] = np.zeros(hidden_size)
        self.params['gamma4'] = np.ones(hidden_size)
        self.params['beta4'] = np.zeros(hidden_size)

        self.params['W5'] = np.random.randn(hidden_size, output_size) * np.sqrt(2.0/hidden_size)
        self.params['b5'] = np.zeros(output_size)

        # --- レイヤ構築 ---
        self.layers = []
        
        # Layer 1
        self.layers.append(Convolution(self.params['W1'], self.params['b1'], conv_param['stride'], conv_param['pad']))
        self.layers.append(BatchNormalization(self.params['gamma1'], self.params['beta1']))
        self.layers.append(Relu())
        self.layers.append(Pooling(pool_h=2, pool_w=2, stride=2))

        # Layer 2
        self.layers.append(Convolution(self.params['W2'], self.params['b2'], conv_param['stride'], conv_param['pad']))
        self.layers.append(BatchNormalization(self.params['gamma2'], self.params['beta2']))
        self.layers.append(Relu())
        self.layers.append(Pooling(pool_h=2, pool_w=2, stride=2))

        # Layer 3
        self.layers.append(Convolution(self.params['W3'], self.params['b3'], conv_param['stride'], conv_param['pad']))
        self.layers.append(BatchNormalization(self.params['gamma3'], self.params['beta3']))
        self.layers.append(Relu())
        # self.layers.append(Pooling(pool_h=2, pool_w=2, stride=2)) # ここで3x3になる

        # Affine 1 (Hidden)
        self.layers.append(Affine(self.params['W4'], self.params['b4']))
        self.layers.append(BatchNormalization(self.params['gamma4'], self.params['beta4']))
        self.layers.append(Relu())

        # Affine 2 (Output)
        self.layers.append(Affine(self.params['W5'], self.params['b5']))
        
        self.last_layer = SoftmaxWithLoss()

    def predict(self, x, train_flg=False):
        if x.ndim == 2:
            x = x.reshape(x.shape[0], 1, 28, 28)
        for layer in self.layers:
            if isinstance(layer, BatchNormalization):
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x, train_flg=True)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t, batch_size=100):
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        acc = 0.0
        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i*batch_size:(i+1)*batch_size]
            tt = t[i*batch_size:(i+1)*batch_size]
            y = self.predict(tx, train_flg=False)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt) 
        return acc / x.shape[0]

    def gradient(self, x, t):
        self.loss(x, t)
        dout = 1
        dout = self.last_layer.backward(dout)
        layers = list(self.layers)
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
        
        grads = {}
        # Conv1 (変更なし)
        grads['W1'], grads['b1'] = self.layers[0].dW, self.layers[0].db
        grads['gamma1'], grads['beta1'] = self.layers[1].dgamma, self.layers[1].dbeta
        
        # Conv2 (変更なし)
        grads['W2'], grads['b2'] = self.layers[4].dW, self.layers[4].db
        grads['gamma2'], grads['beta2'] = self.layers[5].dgamma, self.layers[5].dbeta
        
        # Conv3 (変更なし)
        grads['W3'], grads['b3'] = self.layers[8].dW, self.layers[8].db
        grads['gamma3'], grads['beta3'] = self.layers[9].dgamma, self.layers[9].dbeta
        
        # --- ここから修正 (インデックスを -1 する) ---
        
        # Affine1 (Hidden)
        # 元 layers[12] -> layers[11]
        grads['W4'], grads['b4'] = self.layers[11].dW, self.layers[11].db
        # 元 layers[13] -> layers[12]
        grads['gamma4'], grads['beta4'] = self.layers[12].dgamma, self.layers[12].dbeta
        
        # Affine2 (Output)
        # 元 layers[15] -> layers[14] (Reluなども1つズレているため、Affine2もズレる)
        # 構成: [11]Affine1 -> [12]BN -> [13]Relu -> [14]Affine2
        grads['W5'], grads['b5'] = self.layers[14].dW, self.layers[14].db
        
        return grads

# ==========================================
# 4. Optimizer (Momentum)
# ==========================================
class Momentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None
    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)
        for key in params.keys():
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            params[key] += self.v[key]

# ==========================================
# 5. データ拡張 (アグレッシブ版: 回転±20度採用)
# ==========================================
def augment_data_aggressive(x, t):
    print("データ拡張(アグレッシブ版)を実行中...")
    print("方針: 既存のシフトに加え、±15〜20度の回転を全データに適用します")
    
    N = x.shape[0]
    x_img = x.reshape(N, 1, 28, 28) 
    x_aug = [x_img] # オリジナル
    t_aug = [t]     # オリジナルラベル

    # --- 1. シフト拡張 ---
    shifts = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    for dy, dx in shifts:
        x_shifted = np.roll(x_img, shift=(dy, dx), axis=(2, 3))
        if dy > 0: x_shifted[:, :, :dy, :] = 0
        elif dy < 0: x_shifted[:, :, dy:, :] = 0
        if dx > 0: x_shifted[:, :, :, :dx] = 0
        elif dx < 0: x_shifted[:, :, :, dx:] = 0
        x_aug.append(x_shifted)
        t_aug.append(t)

    # --- 2. 回転拡張 (OpenCV使用) ---
    angles = [-20, -15, 15, 20] 

    for angle in angles:
        x_rotated_list = []
        for i in range(N):
            img = x_img[i, 0]
            M = cv2.getRotationMatrix2D((14, 14), angle, 1.0)
            dst = cv2.warpAffine(img, M, (28, 28), flags=cv2.INTER_LINEAR, borderValue=0)
            x_rotated_list.append(dst)
        
        x_rotated_np = np.array(x_rotated_list).reshape(N, 1, 28, 28)
        x_aug.append(x_rotated_np)
        t_aug.append(t)

    return np.concatenate(x_aug), np.concatenate(t_aug)

# パス設定
DATA_PATH = r".\1_data\train\train_data.npy"
LABEL_PATH = r".\1_data\train\train_label.npy"
PICKLE_NAME = r".\pickle\katakana_model.pickle"
PICKLE_END_NAME = r".\pickle\katakana_model_end.pickle"

# ハイパーパラメータ (時間がない場合は Iters=15000, batch_size=100 に戻してください)
#iters_num = 50000
#iters_num = 10000
ITERS_NUM = 1000 
#batch_size = 32
BATCH_SIZE = 100
LEARNING_RATE = 0.01

# ==========================================
# 6. メイン処理
# ==========================================
if not os.path.exists(DATA_PATH):
    print(f"エラー: ファイルが見つかりません {DATA_PATH}")
    exit()
x_train_raw = np.load(DATA_PATH)
if not os.path.exists(LABEL_PATH):
    print(f"エラー: ファイルが見つかりません {LABEL_PATH}")
    exit()
t_train_raw = np.load(LABEL_PATH)

x_train = x_train_raw.reshape(-1, 1, 28, 28)
if x_train.max() > 1.0: x_train = x_train / 255.0

# シャッフル
p = np.random.permutation(x_train.shape[0])
x_train = x_train[p]
t_train_raw = t_train_raw[p]

# 分割
train_size = int(x_train.shape[0] * 0.8)
x_train_split = x_train[:train_size]
t_train_split = t_train_raw[:train_size]
x_val_split = x_train[train_size:]
t_val_split = t_train_raw[train_size:]

# データ拡張 (アグレッシブ版に変更)
x_train_aug, t_train_aug = augment_data_aggressive(x_train_split, t_train_split)
print(f"学習データ数: {x_train_split.shape[0]} -> {x_train_aug.shape[0]}")

# モデルを Three_ConvNet に変更
network = Three_ConvNet(input_dim=(1,28,28), hidden_size=100, output_size=15)
optimizer = Momentum(lr=LEARNING_RATE)

print(f"\n--- Three_ConvNet + BatchNorm 学習開始 (Iters={ITERS_NUM}, Batch={BATCH_SIZE}) ---")
best_acc = 0.0

for i in range(ITERS_NUM):
    # 学習率減衰
    if i == int(ITERS_NUM * 0.5): optimizer.lr *= 0.1
    if i == int(ITERS_NUM * 0.8): optimizer.lr *= 0.1

    batch_mask = np.random.choice(x_train_aug.shape[0], BATCH_SIZE)
    x_batch = x_train_aug[batch_mask]
    t_batch = t_train_aug[batch_mask]
    
    grads = network.gradient(x_batch, t_batch)
    optimizer.update(network.params, grads)
    
    if i % 100 == 0:
        loss = network.loss(x_batch, t_batch)
        print(f"Iter {i:05d} | Loss: {loss:.4f}")

    if i % 1000 == 0: # 頻度を落とす
        val_acc = network.accuracy(x_val_split, t_val_split)
        print(f"=== Validation Acc: {val_acc:.3f} (Best: {best_acc:.3f}) ===")
        # ★ここが追加ポイント: 最高記録を更新したら、即保存！
        if val_acc >= best_acc:
            # "wb" で上書き保存
            with open(PICKLE_NAME, "wb") as f:
                pickle.dump(network, f)
            print(f"   -> ★Best更新！ katakana_model.pickle を保存しました。")
            best_acc = val_acc
            
print("--- 学習完了 ---")
final_acc = network.accuracy(x_val_split, t_val_split)
print(f"最終検証精度: {final_acc*100:.2f}%")

# 保存 (モデル名も変更)
with open(PICKLE_END_NAME, "wb") as f:
    pickle.dump(network, f)
print("モデル保存完了")