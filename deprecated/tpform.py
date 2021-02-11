import mechanize
br = mechanize.Browser()
br.open("https://blackoutunlock.typeform.com/to/a1hZlnO0")
print(br)
br.select_form()