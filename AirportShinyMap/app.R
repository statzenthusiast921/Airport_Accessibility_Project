#Airplane Routes
#Find some accessibility measure
#Last Update: 04/25/2020
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

#Create column that assigns number of unique routes to single airport
fullair3=fullair2 %>%
  group_by(IATA) %>%
  mutate(Count=n_distinct(UniqueID)) %>%
  ungroup()
fullair3=as.data.frame(fullair3)

range(fullair3$Count)
#1    239

#Get rid of duplicates
fullair3=fullair3[!duplicated(fullair3[c("UniqueID")]),]
dim(fullair3)
#37445    10

library(sqldf)
library(RSQLite)

#-------------------Grab Region and/or Country-------------------#

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

fullair3$Name2Part1=paste("(",fullair3$DestinationAirport,")",sep ="")
fullair3$DestFullName=paste(fullair3$DestAirportName, fullair3$Name2Part1)


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
fullair3$Region=ifelse(fullair3$IATA=="PWE","Asia",fullair3$Region)
fullair3$Region=ifelse(fullair3$IATA=="SUK","Asia",fullair3$Region)
fullair3$Region=ifelse(fullair3$IATA=="DEE","Asia",fullair3$Region)
fullair3$Region=ifelse(fullair3$IATA=="EKS","Asia",fullair3$Region)
fullair3$Region=ifelse(fullair3$IATA=="NGK","Asia",fullair3$Region)

#Get Rid of Missing Values After Join
dim(fullair3)
#37445    22

#Sort by FullName
fullair3= fullair3[order(fullair3$FullName),]


#Name: Jon (me)
#Purpose: Making a Shiny dashboard for my airport data
#Date: 04/23/2020



#-----------------------------------------------------
library(shinydashboard)
library(shiny)
library(leaflet)
library(leaflet.extras)
library(rgdal)
library(sp)
library(raster)
library(DT)
library(markdown)
library(geosphere)
library(htmltools)

airportchoices=unique(fullair3$FullName)
countrychoices=unique(fullair3$Country)
regionchoices=unique(fullair3$Region)

countrychoices=as.character(countrychoices)
countrychoices=sort(countrychoices)

regionchoices=as.character(regionchoices)
regionchoices=sort(regionchoices)


# Define UI for application
ui <- fluidPage(
  dashboardPage(
    dashboardHeader(title="Airport Analysis"),
    dashboardSidebar(
      sidebarMenu(
        menuItem("Welcome",
                 tabName = "welcome",
                 icon=icon("door-open")
        ),
        menuItem("All Data",
                 tabName = "AllData",
                 icon=icon("table")
        ),
        menuItem("Region Maps",
                 tabName = "RegionMaps",
                 icon=icon("globe")
        ),
        menuItem("Country Maps",
                 tabName = "CountryMaps",
                 icon=icon("flag")
        ),
        menuItem("Destination Maps",
                 tabName = "DestMaps",
                 icon=icon("plane-arrival")
        ))
    ),
    dashboardBody(
      tabItems(
        tabItem(
          tabName = "welcome",
          fluidPage(h1("Welcome to my Airport Accessibility Shiny Dashboard!"),
                    br(),
                    p(strong(tags$u("Why did I conduct this analysis?"))),
                    p("I wanted to visualize the accessibility of each city/town/village in the world with an airport."),
                    br(),
                    p(strong(tags$u("What data did I use?"))),
                    p("I used two datasets from the",a("OpenFlights",href="https://openflights.org/data.html"),"code base on Github."),
                    p("1. Dataset consisting of all possible routes from each airport in the world."),
                    p("2. Dataset consisting of descriptive information for each airport in the world."),
                    br(),
                    p(strong(tags$u("Anything else we should know about this data?"))),
                    p("These datasets are not exhaustive lists of every airport in the world.  Further, this data has not been updated since 2014, so certain routes that are currently available may not be reflected in this analysis.")
                    
          )),
        tabItem(
          tabName = "AllData",
          DTOutput("alldata")
        ),
        tabItem(
          tabName = "RegionMaps",
          tags$style(type="text/css","#region_airports {height:calc(100vh - 80px) !important;}"),
          fluidRow(column(4),
                   column(8, selectInput(inputId = "regionselect",label="Select a region:",choices=regionchoices)
                   )),
          DTOutput("regiondata"),
          leafletOutput("region_airports")
        ),
        tabItem(
          tabName = "CountryMaps",
          tags$style(type="text/css","#country_airports {height:calc(100vh - 80px) !important;}"),
          fluidRow(column(4),
                   column(8, selectInput(inputId = "countryselect",label="Select a country:",choices=countrychoices)
                   )),
          DTOutput("countrydata"),
          leafletOutput("country_airports")
          
        ),
        
        tabItem(
          tabName = "DestMaps",
          tags$style(type="text/css","#dest_airports {height:calc(100vh - 80px) !important;}"),
          fluidRow(column(4),
                   column(8, selectInput(inputId = "destselect",label="Select an airport:",choices=airportchoices)
                   )),
          DTOutput("destdata"),
          leafletOutput("dest_airports")
        )
      )
    )
  )
)
# Define server logic 
server <- function(input, output) {
  
  #----------REGION FILTERING---------#  
  RegionData=reactive({
    filteredData=subset(fullair3,Region == input$regionselect)
    return(filteredData)
  })
  
  RegionDataTable=reactive({
    filteredDataTable=subset(table1,Region==input$regionselect)
    filteredDataTable$Region=NULL
    return(filteredDataTable)
  })
  #----------COUNTRY FILTERING---------#  
  
  CountryData=reactive({
    filteredData=subset(fullair3,Country == input$countryselect)
    return(filteredData)
  })
  
  CountryDataTable=reactive({
    filteredDataTable=subset(table2,Country==input$countryselect)
    filteredDataTable$Country=NULL
    return(filteredDataTable)
  })
  
  #----------DESTINATION FILTERING---------#  
  
  AirportData=reactive({
    filteredData=subset(fullair3, FullName == input$destselect)
    filteredData=filteredData[complete.cases(filteredData), ]
    return(filteredData)
  })
  
  AirportDataTable=reactive({
    filteredDataTable=subset(table3,FullName==input$destselect)
    filteredDataTable$FullName=NULL
    filteredDataTable=filteredDataTable[complete.cases(filteredDataTable), ]
    return(filteredDataTable)
  })
  
  #----------ALL DATA TABLE----------#
  output$alldata=renderDT({
    
    datatable(table4,options=list(pageLength=10,
                                  lengthMenu=c(10,15,20,25)
    ),rownames = FALSE)
  })
  
  
  #-------------------REGION PLOTS-------------------#
  
  output$regiondata=renderDT({
    
    data_table = RegionDataTable()
    
    datatable(data_table,options=list(pageLength=5,
                                      lengthMenu=c(5,10,15,20)
    ),rownames = FALSE)
  })
  
  output$region_airports=renderLeaflet({
    
    data=RegionData()
    
    data <- data %>% 
      dplyr::distinct(IATA, Latitude, Longitude, DestinationCount, Name, City) %>% 
      dplyr::arrange(desc(DestinationCount))
    
    pal=colorNumeric("Yellow",data$DestinationCount)
    
    
    leaflet(data=data) %>% 
      addTiles(group="CartoDB.Positron")  %>%
      
      addProviderTiles(providers$CartoDB.Positron,
                       options = tileOptions(minZoom =0, maxZoom = 13),
                       group = "CartoDB.Positron") %>%
      
      addProviderTiles(providers$CartoDB.DarkMatter,
                       options = tileOptions(minZoom =0, maxZoom = 13),
                       group = "CartoDB.DarkMatter") %>%
      
      addProviderTiles(providers$Esri.WorldImagery,
                       options = tileOptions(minZoom =0, maxZoom = 13),
                       group = "Esri.WorldImagery") %>%
      
      
      addCircles(radius = ~data$DestinationCount*1000, 
                 weight = 1, 
                 color = "black", 
                 fillColor = ~pal(data$DestinationCount),
                 fillOpacity = 0.62,
                 popup = paste0("Airport Name: ", data$Name, "<br>",
                                "City: ", data$City, "<br>",
                                "Destination Count: ",data$DestinationCount,"<br>"
                 ),
                 label = ~as.character(data$IATA),
                 group = "Points") %>%
      
      addLayersControl(
        baseGroups = c("CartoDB.Positron","CartoDB.DarkMatter","Esri.WorldImagery"),
        options = layersControlOptions(collapsed = TRUE))
  })
  
  
  
  #-------------------COUNTRY PLOTS-------------------#
  output$countrydata=renderDT({
    
    data_table = CountryDataTable()
    
    
    
    datatable(data_table,options=list(pageLength=5,
                                      lengthMenu=c(5,10,15,20)
    ),rownames = FALSE)
  })
  
  output$country_airports=renderLeaflet({
    
    data=CountryData()
    
    data <- data %>% 
      dplyr::distinct(IATA, Latitude, Longitude, DestinationCount, Name, City) %>% 
      dplyr::arrange(desc(DestinationCount))
    
    
    pal=colorNumeric("Yellow",data$DestinationCount)
    
    
    leaflet(data=data) %>% 
      addTiles(group="CartoDB.Positron")  %>%
      
      addProviderTiles(providers$CartoDB.Positron,
                       options = tileOptions(minZoom =0, maxZoom = 13),
                       group = "CartoDB.Positron") %>%
      
      addProviderTiles(providers$CartoDB.DarkMatter,
                       options = tileOptions(minZoom =0, maxZoom = 13),
                       group = "CartoDB.DarkMatter") %>%
      
      addProviderTiles(providers$Esri.WorldImagery,
                       options = tileOptions(minZoom =0, maxZoom = 13),
                       group = "Esri.WorldImagery") %>%
      
      
      
      addCircles(radius = ~data$DestinationCount*1000, 
                 weight = 1, 
                 color = "black", 
                 fillColor = ~pal(data$DestinationCount),
                 fillOpacity = 0.62,
                 popup = paste0("Airport Name: ", data$Name, "<br>",
                                "City: ", data$City, "<br>",
                                "Destination Count: ",data$DestinationCount,"<br>"
                 ),
                 label = ~as.character(data$IATA),
                 group = "Points") %>%
      
      addLayersControl(
        baseGroups = c("CartoDB.Positron","CartoDB.DarkMatter","Esri.WorldImagery"),
        options = layersControlOptions(collapsed = TRUE))
    
  })
  
  
  
  #-------------------DESTINATION PLOTS-------------------#
  
  
  
  output$destdata=renderDT({
    
    data_table = AirportDataTable()
    
    datatable(data_table,options=list(pageLength=5,
                                      lengthMenu=c(5,10,15,20)
    ),rownames = FALSE)
  })
  
  output$dest_airports=renderLeaflet({
    
    data=AirportData()
    
   # data <- data %>% 
     # dplyr::distinct(IATA, Latitude, Longitude, DestinationCount, Name, City) %>% 
     # dplyr::arrange(desc(DestinationCount))
    
    
    pal=colorNumeric("Yellow",data$DestCount)
    
    
    curved.lines2 =
      gcIntermediate(
        p1 = as.matrix(x = data[,c("Longitude","Latitude")],na.rm=TRUE ),
        p2 = as.matrix(x = data[,c("DestLong","DestLat")],na.rm=TRUE ),
        breakAtDateLine = TRUE,
        n = 1000,
        addStartEnd = TRUE,
        sp = TRUE)
    
    leaflet(data=data) %>% 
      
      
      
      
      addTiles(group="CartoDB.Positron")  %>%
      
      
      addProviderTiles(providers$CartoDB.Positron,
                       options = tileOptions(minZoom =0, maxZoom = 6),
                       group = "CartoDB.Positron") %>%
      
      addProviderTiles(providers$CartoDB.DarkMatter,
                       options = tileOptions(minZoom =0, maxZoom = 6),
                       group = "CartoDB.DarkMatter") %>%
      
      addProviderTiles(providers$Esri.WorldImagery,
                       options = tileOptions(minZoom =0, maxZoom = 6),
                       group = "Esri.WorldImagery") %>%
      
      
      addCircleMarkers(lng = ~data$Longitude,
                       lat = ~data$Latitude,
                       radius = 2,
                       color = "red",
                       label = paste(data$FullName)) %>%
      
      addCircleMarkers(lng = data$DestLong,
                       lat = data$DestLat,
                       radius = ~data$DestCount/25, 
                       color = "black", 
                       fillColor = ~pal(data$DestCount),
                       fillOpacity = 0.62,
                       weight=1,
                       popup = paste0("Airport Name: ", data$DestAirportName, "<br>",
                                      "City: ", data$DestCity, "<br>",
                                      "Destination Count: ",data$DestCount,"<br>"
                       ),
                       label = ~as.character(data$DestinationAirport),
                       group = "Points") %>%
      
      addPolylines( data = curved.lines2, weight = 1,
                    group="Routes") %>%
      
      addLayersControl(
        baseGroups = c("CartoDB.Positron","CartoDB.DarkMatter","Esri.WorldImagery"),
        overlayGroups = c("Routes","Points"),
        options = layersControlOptions(collapsed = TRUE)
      )
    
  })
  
  
  
  
}

# Run the application 
shinyApp(ui = ui, server = server)

