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
    NotImplementedError
  end
end
