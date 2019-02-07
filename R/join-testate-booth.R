join_testate_booth <- function(y) {
  ## Join taxanomic groups for consistency

  names(y)

  y[, names(y) == "arccat"] <- y[, names(y) == "arccat"] + y[, names(y) == "arcart"]
  y <- subset(y, select=-c(arcart))

  y[, names(y) == "arcvul"] <-  y[, names(y) == "arccre"] + y[, names(y) == "arcgib"] +
    y[, names(y) == "archem"] + y[, names(y) == "arcvul"]
  y <- subset(y, select=-c(arccre, arcgib, archem))

  y[, names(y) == "cencas"] <- y[, names(y) == "cencas"] + y[, names(y) == "cenpla"]
  y <- subset(y, select=-c(cenpla))

  # y[, names(y) == "cycarc"] <- y[, names(y) == "cycarc"] + y[, names(y) == "cycxxx"]
  # y <- subset(y, select=-c(cycxxx))

  y[, names(y) == "difacu"] <- y[, names(y) == "difacu"] + y[, names(y) == "difbac"]
  y <- subset(y, select=-c(difbac))

  y[, names(y) == "difobl"] <- y[, names(y) == "difbaf"] + y[, names(y) == "difobl"]
  y <- subset(y, select=-c(difbaf))

  y[, names(y) == "diflei"] <-  y[, names(y) == "difluc"] + y[, names(y) == "diflei"] +
    y[, names(y) == "difovi"] + y[, names(y) == "difrub"] +
    y[, names(y) == "difpri"] #+ y[, names(y) == "difund"]
  y <- subset(y, select=-c(difluc, difovi, difrub, difpri))
  # y <- subset(y, select=-c(difluc, difovi, difrub, difpri, difund))

  y[, names(y) == "difglo"] <- y[, names(y) == "difglo"] + y[, names(y) == "difurc"]
  y <- subset(y, select=-c(difurc))

  y[, names(y) == "eugstr"] <- y[, names(y) == "eugstr"] + y[, names(y) == "eugcri"]
  y <- subset(y, select=-c(eugcri))

  y[, names(y) == "eugrot"] <- y[, names(y) == "eugtub"] + y[, names(y) == "eugrot"]
  y <- subset(y, select=-c(eugtub))

  y[, names(y) == "helpet"] <- y[, names(y) == "helpet"] + y[, names(y) == "helros"]
  y <- subset(y, select=-c(helros))

  y[, names(y) == "nebcar"] <- y[, names(y) == "nebcar"] + y[, names(y) == "nebmar"]
  y <- subset(y, select=-c(nebmar))

  y[, names(y) == "nebtub"] <- y[, names(y) == "nebwai"] + y[, names(y) == "nebtub"]
  y <- subset(y, select=-c(nebwai))

  return(y)
}
