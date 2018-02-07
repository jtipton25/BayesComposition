## Join taxanomic groups for consistency

names(y_recon)

y_recon[, names(y_recon) == "arccat"] <- y_recon[, names(y_recon) == "arccat"] + y_recon[, names(y_recon) == "arcart"]
y_recon <- subset(y_recon, select=-c(arcart))

# y_recon[, names(y_recon) == "arcvul"] <-  y_recon[, names(y_recon) == "arccre"] + y_recon[, names(y_recon) == "arcgib"] +
#  y_recon[, names(y_recon) == "archem"] + y_recon[, names(y_recon) == "arcvul"]
# y_recon <- subset(y_recon, select=-c(arccre, arcgib, archem))

#y_recon[, names(y_recon) == "arcvul"] <-  y_recon[, names(y_recon) == "arcgib"] 
#y_recon <- subset(y_recon, select=-c(arcgib))

y_recon[, names(y_recon) == "cencas"] <- y_recon[, names(y_recon) == "cencas"] + y_recon[, names(y_recon) == "cenpla"]
y_recon <- subset(y_recon, select=-c(cenpla))

# y_recon[, names(y_recon) == "cycarc"] <- y_recon[, names(y_recon) == "cycarc"] + y_recon[, names(y_recon) == "cycxxx"]
# y_recon <- subset(y_recon, select=-c(cycxxx))

y_recon[, names(y_recon) == "difacu"] <- y_recon[, names(y_recon) == "difacu"] + y_recon[, names(y_recon) == "difbum"]
y_recon <- subset(y_recon, select=-c(difbum))

y_recon[, names(y_recon) == "difobl"] <- y_recon[, names(y_recon) == "difbra"] + y_recon[, names(y_recon) == "difobl"]
y_recon <- subset(y_recon, select=-c(difbra))

y_recon[, names(y_recon) == "diflei"] <-  y_recon[, names(y_recon) == "difluc"] + y_recon[, names(y_recon) == "diflei"] +
                              # y_recon[, names(y_recon) == "difovi"] 
                              +  y_recon[, names(y_recon) == "difrub"] +
                              y_recon[, names(y_recon) == "difpri"] #+ y_recon[, names(y_recon) == "difund"]
#y_recon <- subset(y_recon, select=-c(difluc, difovi, difrub, difpri))
y_recon <- subset(y_recon, select=-c(difluc, difpri))
# y_recon <- subset(y_recon, select=-c(difluc, difovi, difrub, difpri, difund))

# y_recon[, names(y_recon) == "difglo"] <- y_recon[, names(y_recon) == "difglo"] + y_recon[, names(y_recon) == "difurc"]
# y_recon <- subset(y_recon, select=-c(difurc))

# y_recon[, names(y_recon) == "eugstr"] <- y_recon[, names(y_recon) == "eugstr"] + y_recon[, names(y_recon) == "eugcri"]
# y_recon <- subset(y_recon, select=-c(eugcri))

# y_recon[, names(y_recon) == "eugrot"] <- y_recon[, names(y_recon) == "eugtub"] + y_recon[, names(y_recon) == "eugrot"]
# y_recon <- subset(y_recon, select=-c(eugtub))

y_recon[, names(y_recon) == "helpet"] <- y_recon[, names(y_recon) == "helpet"] + y_recon[, names(y_recon) == "helros"]
y_recon <- subset(y_recon, select=-c(helros))

# y_recon[, names(y_recon) == "nebtub"] <- y_recon[, names(y_recon) == "nebwai"] + y_recon[, names(y_recon) == "nebtub"]
# y_recon <- subset(y_recon, select=-c(nebwai))

# y_recon[, names(y_recon) == "cortri"] <- y_recon[, names(y_recon) == "cortri"] + y_recon[, names(y_recon) == "cordub"]
# y_recon <- subset(y_recon, select=-c(cordub))

y_recon[, names(y_recon) == "triarc"] <- y_recon[, names(y_recon) == "triarc"] + y_recon[, names(y_recon) == "trimin"]
y_recon <- subset(y_recon, select=-c(trimin))