require_relative 'mlp'
require_relative 'dataset'

# --------------- データセットの準備 -----------------
iris_dataset = Dataset.get_iris

# データセットを分割する
train_size, valid_size = 100, 30
train, other = Chainer::Datasets.split_dataset_random(iris_dataset, train_size)
valid, test = Chainer::Datasets.split_dataset_random(other, valid_size)

# --------------- イテレータの準備 -----------------
batch_size = 4
train_iter = Chainer::Iterators::SerialIterator.new(train, batch_size)
valid_iter = Chainer::Iterators::SerialIterator.new(valid, batch_size, repeat: false, shuffle: false)

# --------------- ネットワークの準備 -----------------
predictor = MLP.new(hidden_nodes_size: 100, output_size: 3)

# --------------- アップデータの準備 -----------------
model = Chainer::Links::Model::Classifier.new(predictor)

optimizer = Chainer::Optimizers::MomentumSGD.new(lr: 0.01)
optimizer.setup(model)

updater = Chainer::Training::StandardUpdater.new(train_iter, optimizer, device: -1) # device=-1でCPUでの計算実行を指定

# --------------- トレーナの作成 -----------------
output_dir = "results/iris_result_#{Time.now.strftime('%Y%m%d_%H%M%S')}"
epoch_size = 30
trainer = Chainer::Training::Trainer.new(updater, stop_trigger: [epoch_size, 'epoch'], out: output_dir)

# --------------- トレーナの拡張 -----------------
Extensions = Chainer::Training::Extensions
trainer.extend(Extensions::Evaluator.new(valid_iter, model, device: -1), name: 'val')
trainer.extend(Extensions::LogReport.new(trigger: [1, 'epoch'], log_name: 'log'))

filename_proc = Proc.new { |t| format('snapshot_epoch-%02d', t.updater.epoch) }
trainer.extend(Extensions::Snapshot.new(filename_proc: filename_proc), trigger: [1, 'epoch'])

entries = %w[epoch iteration main/loss main/accuracy val/main/loss val/main/accuracy elapsed_time]
trainer.extend(Extensions::PrintReport.new(entries))
trainer.extend(Extensions::ProgressBar.new)

# --------------- 訓練の開始 -----------------
trainer.run

# --------------- 推論 -----------------
predictor = MLP.new(hidden_nodes_size: 100, output_size: 3)

snapshot_filename = "#{output_dir}/#{format('snapshot_epoch-%02d', epoch_size)}"
path = '/updater/model:main/@predictor/'
Chainer::Serializers::MarshalDeserializer.load_file(snapshot_filename, predictor, path: path)

print '-' * 100 + "\n"

pass_count = 0
(0...test.size).each do |i|
  variables, answer = test[i]

  # 変数をモデルに与えて推論結果を取得
  prediction = predictor.(variables).data.argmax
  pass_count += 1 if prediction == answer

  print format("test%03d: prediction = %d, answer = %d\n",i, prediction, answer)
end

print "accuracy: #{pass_count * 100.0 / test.size}\n"
