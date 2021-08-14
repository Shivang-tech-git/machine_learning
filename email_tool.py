from exchangelib import DELEGATE, Account, Credentials, Configuration

creds = Credentials(
    username="R01-784111254C.R01.XLGS.LOCAL\A134391",
    password="355008@Axk"
)

config = Configuration(server='mail.mte.xlcatlin.com', credentials=creds)

account = Account(
    primary_smtp_address="shivang.gupta@axaxl.com",
    autodiscover=False,
    config=config,
      access_type=DELEGATE
)

for item in account.inbox.all().order_by('-datetime_received')[:100]:
    print(item.subject, item.sender, item.datetime_received)