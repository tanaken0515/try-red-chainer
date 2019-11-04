require 'datasets'
require 'chainer'

module Dataset
  def self.get_iris(with_label: true)
    raw_variables, raw_labels = retrieve_iris
    variables, labels = preprocess_iris(raw_variables, raw_labels)

    with_label ? Chainer::Datasets::TupleDataset.new(variables, labels) : variables
  end

  def self.preprocess_iris(raw_variables, raw_labels)
    variables = raw_variables.transpose

    label_index_array = raw_labels.uniq
    labels = raw_labels.map do |label|
      label_index_array.index(label)
    end

    xm = Chainer::Device.default.xm

    [xm::SFloat.cast(variables), xm::Int32.cast(labels)]
  end

  def self.retrieve_iris
    raw = Datasets::Iris.new
    raw_table = raw.to_table
    variables = raw_table.fetch_values(:sepal_length, :sepal_width, :petal_length, :petal_width)
    labels = raw_table[:label]

    [variables, labels]
  end
end
