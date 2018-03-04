require "csv"
require "json"

QUESTION_LIMIT = 100_000

language_tags = JSON.parse File.read("languages.json")
questions = CSV.read "raw_questions.csv"

questions.shift

questions.each do |question|
  question_tags = JSON.parse question[2].gsub("'", '"')
  question_tags.select! { |tag| language_tags.include? tag }

  question[2] = question_tags.one? ? question_tags.first : nil
  question[3] = question[3].gsub(/&.*?;/, '')
end

questions.select!(&:all?)
questions.shuffle!

questions = questions.first(QUESTION_LIMIT)

CSV.open("questions.csv", "w") do |csv|
  questions.each { |question| csv << question }
end
