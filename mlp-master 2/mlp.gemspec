# Generated by jeweler
# DO NOT EDIT THIS FILE
# Instead, edit Jeweler::Tasks in Rakefile, and run `rake gemspec`
# -*- encoding: utf-8 -*-

Gem::Specification.new do |s|
  s.name = %q{mlp}
  s.version = "0.0.0"

  s.required_rubygems_version = Gem::Requirement.new(">= 0") if s.respond_to? :required_rubygems_version=
  s.authors = ["reddavis"]
  s.date = %q{2009-09-02}
  s.description = %q{Multi-Layer Perceptron Neural Network in Ruby}
  s.email = %q{reddavis@gmail.com}
  s.extra_rdoc_files = [
    "LICENSE",
     "README.rdoc"
  ]
  s.files = [
    ".autotest",
     ".document",
     ".gitignore",
     "LICENSE",
     "README.rdoc",
     "Rakefile",
     "VERSION",
     "examples/backpropagation_example.rb",
     "examples/patterns_with_base_noise.rb",
     "examples/patterns_with_noise.rb",
     "examples/training_patterns.rb",
     "examples/xor.rb",
     "lib/mlp.rb",
     "lib/neuron.rb",
     "mlp.gemspec",
     "test/helper.rb",
     "test/test_mlp.rb",
     "test/test_neuron.rb"
  ]
  s.homepage = %q{http://github.com/reddavis/mlp}
  s.rdoc_options = ["--charset=UTF-8"]
  s.require_paths = ["lib"]
  s.rubygems_version = %q{1.3.5}
  s.summary = %q{Multi-Layer Perceptron Neural Network in Ruby}
  s.test_files = [
    "test/helper.rb",
     "test/test_mlp.rb",
     "test/test_neuron.rb",
     "examples/backpropagation_example.rb",
     "examples/patterns_with_base_noise.rb",
     "examples/patterns_with_noise.rb",
     "examples/training_patterns.rb",
     "examples/xor.rb"
  ]

  if s.respond_to? :specification_version then
    current_version = Gem::Specification::CURRENT_SPECIFICATION_VERSION
    s.specification_version = 3

    if Gem::Version.new(Gem::RubyGemsVersion) >= Gem::Version.new('1.2.0') then
    else
    end
  else
  end
end
