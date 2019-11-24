require_relative 'mlp'
require_relative 'dataset'
require 'optparse'

snapshot_filename = nil
opt = OptionParser.new
opt.on('-r', '--resume VALUE', "Resume the training from snapshot") { |v| snapshot_filename = v }
opt.parse!(ARGV)

# 学習させた時と同じモデルを定義
predictor = MLP.new(hidden_nodes_size: 100, output_size: 3)

# スナップショットをモデルに読み込む
Chainer::Serializers::MarshalDeserializer.load_file(snapshot_filename, predictor, path: '/updater/model:main/@predictor/')

# テスト用のデータセットを取得
test_dataset = Dataset.get_iris

pass_count = 0
(0...test_dataset.size).each do |i|
  variables, answer = test_dataset[i]

  # 変数をモデルに与えて推論結果を取得
  prediction = predictor.(variables).data.argmax
  pass_count += 1 if prediction == answer

  print format("test%03d: prediction = %d, answer = %d\n",i, prediction, answer)
end

print "accuracy: #{pass_count * 100.0 / test_dataset.size}\n"
