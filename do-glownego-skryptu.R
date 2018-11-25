# algorytm genetyczny do konsoli wypluje ci??g 24 zer lub jedynek oddzielonych spacjami
# trzeba w g????wnym uruchomieniu programu zastosowa?? ten wynik
# w tym celu nale??y do myResult wklei?? ten ci??g (na pocz??tku i ko??cu bez spacji)
# i wklei?? ca??y ten skrypt do pod linijk?? gdzie usuwali??my project, wersje itp
myResult= ""
sss = strsplit(myResult, "\\s+")[[1]]
sss[1]

columns = c("wmc","dit","noc","cbo","rfc","lcom","ca","ce","npm","lcom3","loc","dam","moa",
            "mfa","cam","ic","cbm","amc","nr","ndc","nml","ndpv","max.cc.","avg.cc.")

for (b in 1:24) {
  if (sss[b]=="1") {
    combined <- combined %>% select(-c(columns[b]))
  }
}
