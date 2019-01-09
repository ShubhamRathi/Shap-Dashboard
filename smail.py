def send_mail(SUBJECT):
	import smtplib
	server = smtplib.SMTP('smtp.gmail.com', 587)
	server.starttls()
	#Next, log in to the server
	server.login("rathishubham1103", "Duucatibike123%")
	FROM= "script@shap-dashboard.com"
	SUBJECT= "Script Updates"
	TEXT="Testing"
	TO=["shubhamiiitbackup@gmail.com"]
	#Send the mail
	msg = """From: %s\nTo: %s\nSubject: %s\n\n%s
	""" % (FROM, ", ".join(TO), SUBJECT, TEXT)
	server.sendmail("rathishubham1103@gmail.com", "shubhamiiitbackup@gmail.com", msg)
	server.quit()

send_mail("Testing")