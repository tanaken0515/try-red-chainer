require 'chainer'
require_relative 'data/dataset'
require_relative 'utility/vocabulary'
require_relative 'translator'

# --------------- データセットの準備 -----------------
dataset_size = 2500
dataset = Dataset.get_jpn2eng(dataset_size)

# データセットを分割する
train, test = Chainer::Datasets.split_dataset_random(dataset, (dataset.size * 0.7).to_i)
# train, valid = Chainer::Datasets.split_dataset_random(train_and_valid, (train_and_valid.size * 0.7).to_i)

# --------------- イテレータの準備 -----------------
# batch_size = 64
# train_iter = Chainer::Iterators::SerialIterator.new(train, batch_size)
# valid_iter = Chainer::Iterators::SerialIterator.new(valid, batch_size, repeat: false, shuffle: false)

# --------------- ネットワークの準備 -----------------
jpn_vocabulary = Examples::Seq2seq::Utility::Vocabulary.from_csv('examples/seq2seq/data/jpn_dictionary.csv')
eng_vocabulary = Examples::Seq2seq::Utility::Vocabulary.from_csv('examples/seq2seq/data/eng_dictionary.csv')
embed_size = 100
model = Translator.new(jpn_vocabulary, eng_vocabulary, embed_size) # todo

epoch_num = 100
output_dir = "results/seq2seq_result_#{Time.now.strftime("%Y%m%d_%H%M%S")}"
(1..epoch_num).each do |epoch|
  print "epoch: #{epoch} / #{epoch_num}, start.\n"

  model.learn(train)
  model_file = "#{output_dir}/#{format('jpn_to_egn_%03d.model', epoch)}"
  model.save_model(model_file)

  print "epoch: #{epoch} / #{epoch_num}, finished.\n"
end

# --------------- アップデータの準備 -----------------
# net = Chainer::Links::Model::Classifier.new(predictor)
#
# optimizer = Chainer::Optimizers::MomentumSGD.new(lr: 0.01)
# optimizer.setup(net)
#
# updater = Chainer::Training::StandardUpdater.new(train_iter, optimizer, device: -1) # device=-1でCPUでの計算実行を指定

# --------------- トレーナの作成 -----------------
# output_dir = "results/iris_result_#{Time.now.strftime("%Y%m%d_%H%M%S")}"
# max_epoch = 30
# trainer = Chainer::Training::Trainer.new(updater, stop_trigger: [max_epoch, 'epoch'], out: output_dir)

# --------------- トレーナの拡張 -----------------
# EXTENSIONS = Chainer::Training::Extensions
# trainer.extend(EXTENSIONS::Evaluator.new(valid_iter, net, device: -1), name: 'val')
# trainer.extend(EXTENSIONS::LogReport.new(trigger: [1, 'epoch'], log_name: 'log'))
#
# filename_proc = Proc.new { |t| format("snapshot_epoch-%02d", t.updater.epoch) }
# trainer.extend(EXTENSIONS::Snapshot.new(filename_proc: filename_proc), trigger: [1, 'epoch'])
#
# trainer.extend(EXTENSIONS::PrintReport.new(['epoch', 'iteration', 'main/loss', 'main/accuracy', 'val/main/loss', 'val/main/accuracy', 'elapsed_time']))
# trainer.extend(EXTENSIONS::ProgressBar.new)

# --------------- 訓練の開始 -----------------
# trainer.run

# --------------- 推論 -----------------
model_for_inference = Translator.new(jpn_vocabulary, eng_vocabulary, embed_size)
model_file = "#{output_dir}/#{format('jpn_to_egn_%03d.model', epoch_num)}"
model_for_inference.load_model(model_file)

print '-' * 100 + "\n"

pass_count = 0
(0...test.size).each do |i|
  jpn_sentence_words, eng_sentence_words = test[i]

  # 変数をモデルに与えて推論結果を取得
  prediction = model_for_inference.inference(jpn_sentence_words)
  pass_count += 1 if prediction == eng_sentence_words

  print format("test%09d\n", i)
  print " - jpn: #{jpn_sentence_words.join(' ')}\n"
  print " - ans: #{eng_sentence_words.join(' ')}\n"
  print " - pre: #{prediction.join(' ')}\n"
  print "\n"
end

print "accuracy: #{pass_count * 100.0 / test.size}\n"


# predictor_for_inference = MLP.new
#
# snapshot_filename = "#{output_dir}/#{format("snapshot_epoch-%02d", max_epoch)}"
# Chainer::Serializers::MarshalDeserializer.load_file(snapshot_filename, predictor_for_inference, path: '/updater/model:main/@predictor/')
#
# print '-' * 100 + "\n"
#
# pass_count = 0
# (0...test.size).each do |i|
#   variables, answer = test[i]
#
#   # 変数をモデルに与えて推論結果を取得
#   prediction = predictor_for_inference.(variables).data.argmax
#   pass_count += 1 if prediction == answer
#
#   print format("test%03d: prediction = %d, answer = %d\n",i, prediction, answer)
# end
#
# print "accuracy: #{pass_count * 100.0 / test.size}\n"
