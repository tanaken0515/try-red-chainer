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
      @hidden = Links::LSTM.new(embed_size, embed_size)
      @w_c1 = Links::Linear.new(embed_size, embed_size)
      @w_c2 = Links::Linear.new(embed_size, embed_size)
      @w_y = Links::Linear.new(embed_size, target_vocab.size)
    end

    @optimizer = Chainer::Optimizers::Adam.new
    @optimizer.setup(self)
  end

  def learn(train_dataset)
    (0...train_dataset.size).each do |i|
      source_sentence_words, target_sentence_words = train_dataset[i]

      @hidden.reset_state
      self.zerograds
      loss = loss(source_sentence_words, target_sentence_words)
      loss.backword
      loss.unchain_backword
      @optimizer.update
    end
  end

  def save_model(model_file)
    NotImplementedError
  end

  def load_model(model_file)
    NotImplementedError
  end

  def inference(source_sentence_words)
    NotImplementedError
  end

  private

  def loss(source_sentence_words, target_sentence_words)
    bar_h_i_list = h_i_list(source_sentence_words)

    unk_and_eos_ids = Numo::Int32[@source_vocab.class::UNK[:id].to_i, @source_vocab.class::EOS[:id].to_i]
    x_i = @embed_x.(Chainer::Variable.new(unk_and_eos_ids))
    h_t = @hidden.(x_i)

    c_t = c_t(bar_h_i_list, h_t.data[0])

    bar_h_t = Chainer::Functions::Activation::Tanh.tanh(@w_c1.(c_t) + @w_c2.(h_t))
    first_wid = @target_vocab.word_to_id(target_sentence_words[0]).to_i
    tx = Chainer::Variable.new(Numo::Int32[first_wid])
    accum_loss = Chainer::Functions::Loss::SoftmaxCrossEntropy.softmax_cross_entropy(@w_y.(bar_h_t), tx)

    (target_sentence_words + [@target_vocab.class::EOS[:word]]).each_cons(2) do |this_word, next_word|
      wid = @target_vocab.word_to_id(this_word).to_i
      y_i = @embed_y.(Chainer::Variable.new(Numo::Int32[wid]))
      h_t = @hidden.(y_i)
      c_t = c_t(bar_h_i_list, h_t.data)

      bar_h_t = Chainer::Functions::Activation::Tanh.tanh(@w_c1.(c_t) + @w_c2.(h_t))
      next_wid = @target_vocab.word_to_id(next_word).to_i
      tx = Chainer::Variable.new(Numo::Int32[next_wid])
      loss = Chainer::Functions::Loss::SoftmaxCrossEntropy.softmax_cross_entropy(@w_y.(bar_h_t), tx)
      accum_loss = accum_loss ? accum_loss + loss : loss
    end

    accum_loss
  end

  def h_i_list(source_sentence_words)
    NotImplementedError
  end

  def c_t(bar_h_i_list, h_t, test=false)
    NotImplementedError
  end
end
