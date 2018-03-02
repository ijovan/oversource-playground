require "csv"
require "json"

language_tags = JSON.parse File.read("../languages.json")
questions = CSV.read "../Pop/output/questions.csv"

questions.shift

questions.each do |question|
  question_tags = JSON.parse question[2].gsub("'", '"')
  question_tags.select! { |tag| language_tags.include? tag }

  question[2] = question_tags.one? ? question_tags.first : nil
  question[3] = question[3].gsub(/&.*?;/, '')
end

questions.select!(&:all?)
questions.shuffle!

CSV.open("../language_questions.csv", "w") do |csv|
  questions.first(100_000).each { |question| csv << question }
end
