#Airplane Routes
#Find some accessibility measure
#Last Update: 03/22/2020
#-------------------------------------
#Load datasets

#Dataset 1: Routes
routes=read.csv(url("https://raw.githubusercontent.com/jpatokal/openflights/master/data/routes.dat"))
dim(routes)
#67662  9
str(routes)

#Dataset #2: Airports
airports=read.csv(url("https://raw.githubusercontent.com/jpatokal/openflights/master/data/airports-extended.dat"))
dim(airports)
#12667 14

head(routes)
head(airports)

#Give Better Names to Columns
colnames(routes)=c("Airline","AirlineID","IATA","SourceAP_ID","DestinationAirport","DestAP_ID","Codeshare","Stops","Equipment")
colnames(airports)=c("AirportID","Name","City","Country","IATA","ICAO","Latitude","Longitude","Altitude","Timezone","DST","TzDatabaseTz","Type","Source")

#Join datasets on Source Airport
#-------------------------------#
#IATA in airports
#SourceAirport in routes (IATA)

fullair=merge(x=routes,y=airports,by="IATA",all.x=TRUE)
dim(routes)
#67662  9
dim(airports)
#12667  14
dim(fullair)
#67662  22


fullair2=subset(fullair,fullair$Type=="airport")
dim(fullair2)
#67467  22


#Clean up dataset
#install.packages("dplyr")
library(dplyr)

#Make a new unique ID by combining IATA and Destination Airport
fullair2$UniqueID=paste0(fullair2$IATA,"_",fullair2$DestinationAirport)
head(fullair2$UniqueID)
length(unique(fullair2$UniqueID))
#37445


fullair2$Source=NULL
fullair2$Type=NULL
fullair2$Codeshare=NULL
fullair2$Equipment=NULL
fullair2$Stops=NULL
fullair2$Airline=NULL
fullair2$AirlineID=NULL
fullair2$SourceAP_ID=NULL
fullair2$DestAP_ID=NULL
fullair2$AirportID=NULL
fullair2$Altitude=NULL
fullair2$Timezone=NULL
fullair2$DST=NULL
fullair2$ICAO=NULL
head(fullair2)

#write.csv(fullair2,"fullair2.csv")

#Create column that assigns number of unique routes to single airport
fullair3=fullair2 %>%
  group_by(IATA) %>%
  mutate(Count=n_distinct(UniqueID)) %>%
  ungroup()
fullair3=as.data.frame(fullair3)

range(fullair3$Count)
#1    239
#write.csv(fullair3,"fullair3.csv")


#Get rid of duplicates
fullair3=fullair3[!duplicated(fullair3[c("UniqueID")]),]
dim(fullair3)
#37445    10

#Let's play around with functions to compare accessibility
#install.packages("rowr")
library(rowr)
#install.packages("sqldf")
library(sqldf)
#install.packages("RSQLite")
library(RSQLite)

#--------------------------------------------#
#Function to grab accessibility measure
accessnum=function(zzz){
SpitOutNum=sqldf(sprintf("select IATA, City, Count(*) as NumRoutes
                          from fullair3
                          where IATA='%s'
                          group by IATA, City",zzz))
return(SpitOutNum)
}
accessnum('STL')
accessnum('CDG')
accessnum('MSN')
accessnum('ORD')
accessnum('DCA')
accessnum('ACV')



#-------------------Grab Continent and/or Country-------------------#

library(stringi)
fullair3$Region=stri_extract(fullair3$TzDatabaseTz, regex='[^/]*')

unique(fullair3$Region)
table(fullair3$Region)


#----------Find out where these "\\N" need to go----------#

#Regional Access Measure
test=subset(fullair3,fullair3$Region=="\\N")
dim(test)
#0    13
sqldf("select Country, count(*) as Count
      from test
      group by Country
      order by Count desc")



#test2=fullair3[fullair3$Country=="Australia",]
#write.csv(test2,"test2.csv")

#Fix 1: Europe Region
fullair3$Region=ifelse(fullair3$Region=="\\N" & (fullair3$Country=="Turkey"|
                                                 fullair3$Country=="Russia"|
                                                 fullair3$Country=="France"),"Europe",fullair3$Region)
#Fix 2: Asia
fullair3$Region=ifelse(fullair3$Region=="\\N" & (fullair3$Country=="Qatar"|
                                                 fullair3$Country=="Pakistan"|
                                                 fullair3$Country=="India"|
                                                 fullair3$Country=="China"|
                                                 fullair3$Country=="Saudi Arabia"|
                                                 fullair3$Country=="Japan"|
                                                 fullair3$Country=="Philippines"|
                                                 fullair3$Country=="Indonesia"),"Asia",fullair3$Region)

#Fix 3: America Region
fullair3$Region=ifelse(fullair3$Region=="\\N" & (fullair3$Country=="Canada"|
                                                 fullair3$Country=="Mexico"|
                                                 fullair3$Country=="Chile"|
                                                 fullair3$Country=="Ecuador"|
                                                 fullair3$Country=="Guadeloupe"|
                                                 fullair3$Country=="Greenland"),"America",fullair3$Region)

#Fix 4: Pacific
fullair3$Region=ifelse(fullair3$Region=="\\N" & (fullair3$Country=="French Polynesia"|
                                                 fullair3$Country=="Kiribati"|
                                                 fullair3$Country=="Papua New Guinea"),"Pacific",fullair3$Region)

#Fix 5: Africa
fullair3$Region=ifelse(fullair3$Region=="\\N" & (fullair3$Country=="Algeria"|
                                                 fullair3$Country=="Uganda"|
                                                 fullair3$Country=="Nigeria"|
                                                 fullair3$Country=="Tanzania"|
                                                 fullair3$Country=="Zambia"|
                                                 fullair3$Country=="Congo (Kinshasa)"|
                                                 fullair3$Country=="Congo (Brazzaville)"),"Africa",fullair3$Region)

#Fix 6: Indian
fullair3$Region=ifelse(fullair3$Region=="\\N" & (fullair3$Country=="Maldives"),"Indian",fullair3$Region)
#Fix 7: Australia
fullair3$Region=ifelse(fullair3$Region=="\\N" & (fullair3$Country=="Australia"),"Australia",fullair3$Region)


#-----------Number of Routes from Destination------------#
SpitOutNum=sqldf("select IATA,count(*)
                          from fullair3
                          group by IATA")
SpitOutNum=as.data.frame(SpitOutNum)
colnames(SpitOutNum)=c("IATA","DestinationCount")
fullair3=merge(x=fullair3,y=SpitOutNum,by="IATA",all.x=TRUE)
dim(fullair3)
#37445    12


#Create the full name
fullair3$NamePart1=paste("(",fullair3$IATA,")",sep ="")
fullair3$FullName=paste(fullair3$Name, fullair3$NamePart1)
fullair3$NamePart1=NULL

fullair3$Name2Part1=paste("(",fullair3$DestinationAirport,")",sep ="")
fullair3$DestFullName=paste(fullair3$DestAirportName, fullair3$Name2Part1)

#Make destination columns
SpitOutNum2=sqldf("select IATA, City, Country, Region, Name, DestinationCount, Longitude, Latitude
                   from fullair3
                   group by IATA,City, Country, Region, Name")
colnames(SpitOutNum2)=c("DestinationAirport","DestCity","DestCountry","DestRegion","DestAirportName","DestCount","DestLong","DestLat")
fullair3=merge(x=fullair3,y=SpitOutNum2,by="DestinationAirport",all.x=TRUE)

#Export the dataset out
#write.csv(fullair3, "AirportAccess.csv")


#Plot some maps
#install.packages(c("cowplot", "googleway", "ggplot2", "ggrepel", 
     #              "ggspatial", "libwgeom", "sf", "rnaturalearth", "rnaturalearthdata"))
library("ggplot2")
theme_set(theme_bw())
library("sf")                 
library("rnaturalearth")
library("rnaturalearthdata")
#install.packages("rgeos")
library(rgeos)


install.packages("ggmap")
library(ggmap)
qmplot(Longitude, Latitude, data = fullair3, colour = I('red'), size = I(3), darken = .3)

Pacific=subset(fullair3,fullair3$Region=="Pacific")
qmplot(Longitude, Latitude, data = Pacific, colour = I('green'), size = I(3), darken = .3)


Italy=subset(fullair3,fullair3$Country=="Italy")
p=qmplot(Longitude, Latitude, data = Italy, colour = I('blue'), size = DestinationCount, darken = .3)
p

Morocco=subset(fullair3,fullair3$Country=="Morocco")
q=qmplot(Longitude, Latitude, data = Morocco, colour = I('blue'), size = DestinationCount, darken = .3)
q

StLouis=subset(fullair3,fullair3$IATA=="STL")


#Let's make P interactive
# load libraries
library(dplyr)
install.packages("rgdal")
library(rgdal)
library(ggplot2)
install.packages("leaflet")
library(leaflet)
# set factors to false
options(stringsAsFactors = FALSE)


map <- leaflet() %>%
  addTiles() %>%  # use the default base map which is OpenStreetMap tiles
  addCircleMarkers(lng = Morocco$Longitude,
                   lat = Morocco$Latitude,
             radius=sqrt(Morocco$DestinationCount)*0.75,
             color="black",fillColor="blue",
             stroke=TRUE,fillOpacity = 0.25,
             popup = paste("Airport Name:", Morocco$Name, "<br>",
                           "Airport Code:",Morocco$IATA,"<br>",
                           "City:",Morocco$City,"<br>",
                           "# Routes:",Morocco$DestinationCount))
map



map2 <- leaflet() %>%
  addTiles() %>%  # use the default base map which is OpenStreetMap tiles
  addCircleMarkers(lng = StLouis$DestLong,
                   lat = StLouis$DestLat,
                   radius=sqrt(StLouis$DestCount)*0.75,
                   color="black",fillColor="blue",
                   stroke=TRUE,fillOpacity = 0.25,
                   popup = paste("Airport Name:", StLouis$DestAirportName, "<br>",
                                 "Airport Code:",StLouis$DestinationAirport,"<br>",
                                 "City:",StLouis$DestCity,"<br>",
                                 "# Routes:",StLouis$DestCount))
map2

#TABLE1 - Region
table1=sqldf("select Region, FullName as 'Airport Name', Country, City, count(*) as 'Number of Routes'
                   from fullair3
      group by Region, FullName, Country, City
      order by count(*) desc")


#TABLE2 - Country
table2=sqldf("select Country, FullName as 'Airport Name', City, count(*) as 'Number of Routes'
             from fullair3
             group by Country, FullName, City
             order by count(*) desc")

#TABLE3 - Destination
table3=sqldf("select FullName, DestFullName as 'Airport Name', DestCountry as 'Country',DestCity as 'City', DestCount as 'Number of Routes'
             from fullair3
             group by FullName, DestFullName, DestCountry, DestCity
             order by DestCount desc")

#TABLE 4 - ALL DATA
table4=sqldf("select FullName as 'Airport Name', Country, City, count(*) as 'Number of Routes'
                   from fullair3
             group by FullName, Country, City
             order by count(*) desc")

#Fix some more issues
#PWE
fullair3$Region=ifelse(fullair3$IATA=="PWE","Asia",fullair3$Region)
fullair3$Region=ifelse(fullair3$IATA=="SUK","Asia",fullair3$Region)
fullair3$Region=ifelse(fullair3$IATA=="DEE","Asia",fullair3$Region)
fullair3$Region=ifelse(fullair3$IATA=="EKS","Asia",fullair3$Region)
fullair3$Region=ifelse(fullair3$IATA=="NGK","Asia",fullair3$Region)



