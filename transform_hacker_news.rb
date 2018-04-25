require "csv"
require "json"

rows = CSV.read("hacker_news.csv")

langs = JSON.parse(File.read("languages.json"))

["c", "r", "go"].each { |lang| langs.delete(lang) }

langs.map! { |lang| " #{lang} " }

filtered_rows = rows.select do |row|
  langs.any? { |lang| row.last.include?(lang) }
end

puts filtered_rows.count

filtered_rows.reject! { |row| row[1].length < 500 }
filtered_rows.each { |row| row[1].gsub!(/\<.*?\>/, '') }
filtered_rows.each { |row| row[1].gsub!(/\&.*?\;/, '') }
filtered_rows.each { |row| row[1].gsub!(/http.*?com/, '') }

puts filtered_rows.count

CSV.open("processed_hacker_news.csv", "wb") do |csv|
  filtered_rows.each { |row| csv << row }
end
