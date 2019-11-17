require 'chainer'

class Translator < Chainer::Chain
  Links = Chainer::Links::Connection

  def initialize(source_vocab, target_vocab, embed_size)
    @source_vocab = source_vocab
    @target_vocab = target_vocab
    @embed_size = embed_size
    super()
    init_scope do
      @embed_x = Links::EmbedID.new(source_vocab.size, embed_size)
      @embed_y = Links::EmbedID.new(target_vocab.size, embed_size)
      @hidden = Links::LSTM.new(embed_size, out_size: embed_size)
      @w_c1 = Links::Linear.new(embed_size, out_size: embed_size)
      @w_c2 = Links::Linear.new(embed_size, out_size: embed_size)
      @w_y = Links::Linear.new(embed_size, out_size: target_vocab.size)
    end

    @optimizer = Chainer::Optimizers::Adam.new
    @optimizer.setup(self)
  end

  def learn(train_dataset)
    (0...train_dataset.size).each do |i|
      source_sentence_words, target_sentence_words = train_dataset[i]

      @hidden.reset_state
      @optimizer.target.cleargrads
      loss = loss(source_sentence_words, target_sentence_words)
      loss.backword
      loss.unchain_backword
      @optimizer.update
    end
  end

  def save_model(model_file)
    Chainer::Serializers::MarshalDeserializer.save_file(model_file, self)
  end

  def load_model(model_file)
    Chainer::Serializers::MarshalDeserializer.load_file(model_file, self)
  end

  def inference(source_sentence_words)
    NotImplementedError
  end

  private

  def loss(source_sentence_words, target_sentence_words)
    bar_h_i_list = h_i_list(source_sentence_words)

    # 先頭の単語を推定して損失誤差を求める
    eos_id = @source_vocab.class::EOS[:id]
    x_i = @embed_x.(Chainer::Variable.new(Numo::Int32[eos_id]))
    h_t = @hidden.(x_i)
    c_t = c_t(bar_h_i_list, h_t.data)

    bar_h_t = Chainer::Functions::Activation::Tanh.tanh(@w_c1.(c_t) + @w_c2.(h_t))
    first_wid = @target_vocab.word_to_id(target_sentence_words[0])
    tx = Chainer::Variable.new(Numo::Int32[first_wid])
    accum_loss = Chainer::Functions::Loss::SoftmaxCrossEntropy.softmax_cross_entropy(@w_y.(bar_h_t), tx)

    # 2単語目以降を1単語ずつ推定して損失誤差を加算していく
    (target_sentence_words + [@target_vocab.class::EOS[:word]]).each_cons(2) do |this_word, next_word|
      wid = @target_vocab.word_to_id(this_word)
      y_i = @embed_y.(Chainer::Variable.new(Numo::Int32[wid]))
      h_t = @hidden.(y_i)
      c_t = c_t(bar_h_i_list, h_t.data)

      bar_h_t = Chainer::Functions::Activation::Tanh.tanh(@w_c1.(c_t) + @w_c2.(h_t))
      next_wid = @target_vocab.word_to_id(next_word)
      tx = Chainer::Variable.new(Numo::Int32[next_wid])
      loss = Chainer::Functions::Loss::SoftmaxCrossEntropy.softmax_cross_entropy(@w_y.(bar_h_t), tx)
      accum_loss = accum_loss ? accum_loss + loss : loss
    end

    accum_loss
  end

  def h_i_list(source_sentence_words)
    source_sentence_words.each_with_object([]) do |word, result|
      wid = @source_vocab.word_to_id(word)
      x_i = @embed_x.(Chainer::Variable.new(Numo::Int32[wid]))
      h_i = @hidden.(x_i)

      result.append(h_i.data[0].clone) # np.copy(h_i.data[0])
    end
  end

  def c_t(bar_h_i_list, h_t)
    s = bar_h_i_list.inject(0.0) do |result, bar_h_i|
      result + Numo::NMath.exp(h_t.dot(bar_h_i))
    end

    c_t_default = Numo::DFloat.zeros(@embed_size)
    c_t = bar_h_i_list.inject(c_t_default) do |result, bar_h_i|
      alpha_t_i = Numo::NMath.exp(h_t.dot(bar_h_i)) / s
      result + alpha_t_i * bar_h_i
    end

    Chainer::Variable.new(Numo::Float32[c_t])
  end
end
