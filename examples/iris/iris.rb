require_relative 'mlp'
require_relative 'dataset'

# --------------- データセットの準備 -----------------
iris_dataset = Dataset.get_iris

# データセットを分割する
train_and_valid, test = Chainer::Datasets.split_dataset_random(iris_dataset, (iris_dataset.size * 0.7).to_i)
train, valid = Chainer::Datasets.split_dataset_random(train_and_valid, (train_and_valid.size * 0.7).to_i)

# --------------- イテレータの準備 -----------------
batch_size = 4
train_iter = Chainer::Iterators::SerialIterator.new(train, batch_size)
valid_iter = Chainer::Iterators::SerialIterator.new(valid, batch_size, repeat: false, shuffle: false)

# --------------- ネットワークの準備 -----------------
predictor = MLP.new

# --------------- アップデータの準備 -----------------
net = Chainer::Links::Model::Classifier.new(predictor)

optimizer = Chainer::Optimizers::MomentumSGD.new(lr: 0.01)
optimizer.setup(net)

updater = Chainer::Training::StandardUpdater.new(train_iter, optimizer, device: -1) # device=-1でCPUでの計算実行を指定

# --------------- トレーナの作成 -----------------
output_dir = "results/iris_result_#{Time.now.strftime("%Y%m%d_%H%M%S")}"
max_epoch = 30
trainer = Chainer::Training::Trainer.new(updater, stop_trigger: [max_epoch, 'epoch'], out: output_dir)

# --------------- トレーナの拡張 -----------------
EXTENSIONS = Chainer::Training::Extensions
trainer.extend(EXTENSIONS::Evaluator.new(valid_iter, net, device: -1), name: 'val')
trainer.extend(EXTENSIONS::LogReport.new(trigger: [1, 'epoch'], log_name: 'log'))

filename_proc = Proc.new { |t| format("snapshot_epoch-%02d", t.updater.epoch) }
trainer.extend(EXTENSIONS::Snapshot.new(filename_proc: filename_proc), trigger: [1, 'epoch'])

trainer.extend(EXTENSIONS::PrintReport.new(['epoch', 'iteration', 'main/loss', 'main/accuracy', 'val/main/loss', 'val/main/accuracy', 'elapsed_time']))
trainer.extend(EXTENSIONS::ProgressBar.new)

# --------------- 訓練の開始 -----------------
trainer.run

# --------------- 推論 -----------------
predictor_for_inference = MLP.new

snapshot_filename = "#{output_dir}/#{format("snapshot_epoch-%02d", max_epoch)}"
Chainer::Serializers::MarshalDeserializer.load_file(snapshot_filename, predictor_for_inference, path: '/updater/model:main/@predictor/')

print '-' * 100 + "\n"

pass_count = 0
(0...test.size).each do |i|
  variables, answer = test[i]

  # 変数をモデルに与えて推論結果を取得
  prediction = predictor_for_inference.(variables).data.argmax
  pass_count += 1 if prediction == answer

  print format("test%03d: prediction = %d, answer = %d\n",i, prediction, answer)
end

print "accuracy: #{pass_count * 100.0 / test.size}\n"
