require_relative 'mlp'

# --------------- データセットの準備 -----------------

# --------------- イテレータの準備 -----------------
train_iter = nil
valid_iter = nil

# --------------- ネットワークの準備 -----------------
predictor = MLP.new

# --------------- アップデータの準備 -----------------
net = Chainer::Links::Model::Classifier.new(predictor)

optimizer = Chainer::Optimizers::MomentumSGD.new(lr: 0.01)
optimizer.set(net)

updater = Chainer::Training::StandardUpdater.new(train_iter, optimizer, device: -1) # device=-1でCPUでの計算実行を指定

# --------------- トレーナの作成 -----------------
trainer = Chainer::Training::Trainer.new(updater, stop_trigger: [30, 'epoch'], out: 'results/iris_result1')

# --------------- トレーナの拡張 -----------------
EXTENSIONS = Chainer::Training::Extensions
trainer.extend(EXTENSIONS::Evaluator.new(valid_iter, net, device: -1), name: 'val')
trainer.extend(EXTENSIONS::LogReport.new(trigger: [1, 'epoch'], log_name: 'log'))

filename_proc = Proc.new { |t| "snapshot_epoch-#{t.updater.epoch}" }
trainer.extend(EXTENSIONS::Snapshot.new(filename_proc: filename_proc), trigger: [1, 'epoch'])

trainer.extend(EXTENSIONS::PrintReport.new(['epoch', 'iteration', 'main/loss', 'main/accuracy', 'val/main/loss', 'val/main/accuracy', 'l1/W/data/mean', 'elapsed_time']))
trainer.extend(EXTENSIONS::ParameterStatistics(net.predictor.l1, {'mean': np.mean}, report_grads=True))

# --------------- 訓練の開始 -----------------
trainer.run
