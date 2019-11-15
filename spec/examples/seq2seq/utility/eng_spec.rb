require 'spec_helper'
require_relative '../../../../examples/seq2seq/utility/eng'

describe Examples::Seq2seq::Utility::Eng do
  describe 'self.parse' do
    subject { Examples::Seq2seq::Utility::Eng.parse(text) }

    context '単語末尾にピリオドがある場合' do
      let(:text) { 'I am fine. ' }
      it 'ピリオドが分割される' do
        expect(subject).to eq('I am fine . ')
      end
    end

    context '単語中にピリオドがある場合' do
      let(:text) { 'Ruby 2.7.0 Preview 2 Released. ' }
      it '単語中のピリオドは分割されない' do
        expect(subject).to eq('Ruby 2.7.0 Preview 2 Released . ')
      end
    end

    context '連続するピリオドがある場合' do
      let(:text) { 'Oh... so cool. ' }
      it '連続するピリオドを1単語として分割される' do
        expect(subject).to eq('Oh ... so cool . ')
      end
    end
  end
end
