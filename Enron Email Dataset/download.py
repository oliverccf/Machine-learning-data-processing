from urllib import request
import tarfile
import os


print("downloading the Enron dataset (this may take a while)")
print("to check on progress, you can cd up one level, then execute <ls -lthr>")
print("Enron dataset should be last item on the list, along with its current size")
print("download will complete at about 423 MB")


os.chdir(os.path.expanduser("~/DataSet"))


url = "https://www.cs.cmu.edu/~./enron/enron_mail_20150507.tar.gz"
request.urlretrieve(url, filename="./enron_mail_20150507.tar.gz")
print("download complete!")

print("unzipping Enron dataset (this may take a while)")


tfile = tarfile.open("enron_mail_20150507.tar.gz", "r:gz")
tfile.extractall(".")

print("you're ready to go!")
