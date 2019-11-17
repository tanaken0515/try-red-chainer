require 'chainer'

class Translator < Chainer::Chain
  Links = Chainer::Links::Connection

  def initialize(source_vocab, target_vocab, embed_size)
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
    NotImplementedError
  end

  def save_model(model_file)
    NotImplementedError
  end

  def load_model(model_file)
    NotImplementedError
  end

  def inference(jpn_sentence_words)
    NotImplementedError
  end
end
