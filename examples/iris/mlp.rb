require 'chainer'

class MLP < Chainer::Chain
  Linear = Chainer::Links::Connection::Linear
  Relu = Chainer::Functions::Activation::Relu

  def initialize(hidden_nodes_size: 100, output_size: 3)
    super()
    init_scope do
      @l1 = Linear.new(nil, out_size: hidden_nodes_size)
      @l2 = Linear.new(hidden_nodes_size, out_size: hidden_nodes_size)
      @l3 = Linear.new(hidden_nodes_size, out_size: output_size)
    end
  end

  def call(x)
    u1 = @l1.(x)
    h1 = Relu.relu(u1)
    u2 = @l2.(h1)
    h2 = Relu.relu(u2)
    @l3.(h2)
  end
end
