from apyori import apriori

filename = "retail.dat"
Transactions = [i.strip().split() for i in open(filename).readlines()]
apriori_results = list(apriori(Transactions,min_support=0.01,max_length=2))

number=1
for item in range(0, len(apriori_results)):
    current=str(apriori_results[item]).split('=')
    print(str(number)+") Набор:"+(current[1])[11:-11])
    print("Поддержка:"+str(round(float((current[2])[:-20])*100,2))+"%")
    if (str(round(float((current[2])[:-20])*100,2))!=str(round(float((current[6])[:-6]) * 100, 2))):
        print("Достоверность:" + str(round(float((current[6])[:-6]) * 100, 2)) + "%")
    print("\n")
    number=number+1
