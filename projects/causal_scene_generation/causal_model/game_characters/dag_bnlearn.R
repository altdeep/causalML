## Loading required packages.
library(bnlearn)
library(Rgraphviz)
library(gRain)


## Create the dag
game_net <- model2network("[AC][RC][AT|AC][RT|RC][AS|AT:AC][AD|AT:AC][AA|AT:AC][RS|RT:RC][RD|RT:RC][RA|RT:RC][AACT|AS:AD:AA][RRCT|RS:RD:RA:AACT]")

## Plot the dag
graphviz.plot(game_net)

## Create the dag
game_net_full <- model2network("[Actor Character][Reactor Character][Actor Type|Actor Character][Reactor Type|Reactor Character][Actor Strength|Actor Type:Actor Character][Actor Defense|Actor Type:Actor Character][Actor Attack|Actor Type:Actor Character][Reactor Strength|Reactor Type:Reactor Character][Reactor Defense|Reactor Type:Reactor Character][Reactor Attack|Reactor Type:Reactor Character][Action|Actor Strength:Actor Defense:Actor Attack][Reaction|Reactor Strength:Reactor Defense:Reactor Attack:Action][IMG|Action:Reaction]")

## Plot the dag
graphviz.plot(game_net_full, shape ='rectangle' )

# Fit a custom dag with probabilities

# Character Nodes (Actor and Reactor)

cptAC = matrix(c(0.5, 0.5), ncol = 2, dimnames = list(NULL, c("Satyr", "Golem")))
cptRC = matrix(c(0.5, 0.5), ncol = 2, dimnames = list(NULL, c("Satyr", "Golem")))

# Type nodes conditioned on Character (Actor and Reactor)

cptAT = c(0.33, 0.34, 0.33, 0.33, 0.34, 0.33)
dim(cptAT) <- c(3,2)
dimnames(cptAT) <- list("AT"=c("Type1", "Type2", "Type3"), "AC" = c("Satyr", "Golem"))

cptRT = c(0.33, 0.34, 0.33, 0.33, 0.34, 0.33)
dim(cptRT) <- c(3,2)
dimnames(cptRT) <- list("RT"=c("Type1", "Type2", "Type3"), "RC" = c("Satyr", "Golem"))

# Attack, strength and defense conditioned on Type and Character (Actor and Reactor)
cptAA <- c(0.2, 0.8, 0.6, 0.4, 0.8, 0.2, 0.75, 0.25, 0.4, 0.6, 0.9, 0.1)
dim(cptAA) <- c(2,3,2)
dimnames(cptAA) <- list("AA"= c("LOW", "HIGH"), "AT"= c("Type1", "Type2", "Type3"), "AC"= c("Satyr", "Golem"))

cptAD <- c(0.9, 0.1, 0.3, 0.7, 0.6, 0.4, 0.5, 0.5, 0.4, 0.6, 0.6, 0.4)
dim(cptAD) <- c(2,3,2)
dimnames(cptAD) <- list("AD"= c("LOW", "HIGH"), "AT"= c("Type1", "Type2", "Type3"), "AC"= c("Satyr", "Golem"))

cptAS <- c(0.4, 0.6, 0.2, 0.8, 0.5, 0.5, 0.6, 0.4, 0.5, 0.5, 0.8, 0.2)
dim(cptAS) <- c(2,3,2)
dimnames(cptAS) <- list("AS"= c("LOW", "HIGH"), "AT"= c("Type1", "Type2", "Type3"), "AC"= c("Satyr", "Golem"))

cptRA <- c(0.2, 0.8, 0.6, 0.4, 0.8, 0.2, 0.75, 0.25, 0.4, 0.6, 0.9, 0.1)
dim(cptRA) <- c(2,3,2)
dimnames(cptRA) <- list("RA"= c("LOW", "HIGH"), "RT"= c("Type1", "Type2", "Type3"), "RC"= c("Satyr", "Golem"))

cptRD <- c(0.9, 0.1, 0.3, 0.7, 0.6, 0.4, 0.5, 0.5, 0.4, 0.6, 0.6, 0.4)
dim(cptRD) <- c(2,3,2)
dimnames(cptRD) <- list("RD"= c("LOW", "HIGH"), "RT"= c("Type1", "Type2", "Type3"), "RC"= c("Satyr", "Golem"))

cptRS <- c(0.4, 0.6, 0.2, 0.8, 0.5, 0.5, 0.6, 0.4, 0.5, 0.5, 0.8, 0.2)
dim(cptRS) <- c(2,3,2)
dimnames(cptRS) <- list("RS"= c("LOW", "HIGH"), "RT"= c("Type1", "Type2", "Type3"), "RC"=c("Satyr", "Golem"))

# Action conditioned on Actor Strength, Attack and defense

cptAACT <- c(0.1, 0.3, 0.6, 0.3, 0.5, 0.2, 0.3, 0.3, 0.4, 0.5,0.4,0.1,0.1, 0.2, 0.7, 0.4,0.3,0.3, 0.2, 0.4, 0.4, 0.6, 0.3, 0.1)
dim(cptAACT)<- c(3,2,2,2)
dimnames(cptAACT)<- list("AACT"= c("Attack", "Taunt", "Walk"), "AA"= c("LOW", "HIGH"), "AD"= c("LOW", "HIGH"), "AS"= c("LOW", "HIGH"))




cptRRCT <- c(0.5, 0.4, 0.05, 0.05, 0.2, 0.6, 0.1, 0.1, 0.001, 0.001, 0.997, 0.001,0.4, 0.3,0.1, 0.2, 0.1, 0.5, 0.2, 0.2, 0.001, 0.001, 0.99, 0.008,0.1, 0.3, 0.55, 0.05, 0.1, 0.2, 0.65, 0.05, 0.001, 0.001, 0.997, 0.001, 0.3, 0.2, 0.3, 0.2,0.1, 0.3, 0.4, 0.2,0.001, 0.001, 0.99, 0.008,0.3, 0.3, 0.399, 0.001,0.2, 0.4, 0.399, 0.001,0.001, 0.001, 0.997, 0.001,  0.3, 0.4, 0.1, 0.2,0.3, 0.3, 0.1, 0.3,0.001, 0.001, 0.99, 0.008, 0.2, 0.3, 0.49, 0.01,0.2, 0.2, 0.59, 0.01,0.001, 0.001, 0.997, 0.001,  0.2, 0.2, 0.4, 0.2,0.1, 0.1, 0.4, 0.4,0.001, 0.001, 0.99, 0.008)
dim(cptRRCT)<- c(4,2,2,2,3)
dimnames(cptRRCT)<- list("RRCT"=c("Dying", "Hurt", "Idle", "Attack"), "RA"= c("LOW", "HIGH"), "RD"= c("LOW", "HIGH"), "RS"= c("LOW", "HIGH"), "AACT"= c("Attack", "Taunt", "Walk"))


dfit <- custom.fit(game_net, dist = list(AC = cptAC, RC=cptRC, AT= cptAT, RT=cptRT, AA=cptAA, AD=cptAD, AS=cptAS, RA=cptRA, RS=cptRS, RD=cptRD, AACT=cptAACT, RRCT=cptRRCT))
grainObj <- as.grain(dfit)


getinvProb<- function(node, ev, event, evidence) {
  #cpstmt <- paste("cpquery(dfit, ",event, ",", evidence, ")", sep = "")
  grStmt <- paste("querygrain(grainObj, nodes = ", event, ", evidence = ", evidence, ")[['", node, "']]", "[['", ev, "']]", sep="")
  print(grStmt)
  expr <- parse(text= grStmt)
  return(eval(expr))
}

actionNodes <- c("AA", "AD", "AS")
reactionNodes <- c("RA", "RD", "RS")


getActionInvCpt <- function(){
  cpt <- list()
  for(node in actionNodes){
    cpt[[node]] <- normalizeProb(getActCptNode(node))
    #dim(cpt[[node]])<- c(2,3,3,2)
    #dimnames(cpt[[node]])<- list(`node`=  c("LOW", "HIGH"), "AACT"=c("Attack", "Taunt", "Walk"),"AT"= c("Type1", "Type2", "Type3"), "AC"= c("Satyr", "Golem"))
  }
  
  return(cpt)
}

normalizeProb<- function(arr){
  # For every 2 elements, divide each element by its sum.
  idx1<-1
  idx2<-2
  while(idx2 <= length(arr)){
    totalProb<- arr[[idx1]] + arr[[idx2]]
    arr[[idx1]] <- arr[[idx1]]/totalProb
    arr[[idx2]] <- arr[[idx2]]/totalProb
    
    idx1 <- idx1 + 2
    idx2 <- idx2 + 2
  }
  return(arr)
}

getReactionInvCpt <- function(){
  cpt <- list()
  for(node in reactionNodes){
    cpt[[node]] <- normalizeProb(getRctCptNode(node))
    #dim(cpt[[node]])<- c(2,4,3,2)
    #dimnames(cpt[[node]])<- list(`node`=  c("LOW", "HIGH"), "RRCT"=c("Dying", "Hurt", "Idle", "Attack"),"AT"= c("Type1", "Type2", "Type3"), "AC"= c("Satyr", "Golem"))
  }
  
  return(cpt)
}

getProb<- function(node, ev, act, typ, chr){
  event = paste("(", node, "==", "'", ev, "')", sep = "")
  print(event)
  evidence = constructEvidence(act, typ, chr)
  print(evidence)
  return(getinvProb(event, evidence))
}

getActCptNode <- function(node){
  idx <- 1
  probs <- c()
  for(act in evidence1Values){
    for(typ in evidence2Values){
      for(chr in evidence3Values){
        for(ev in eventValues){
          #event = paste("(", node, "==", "'", ev, "')", sep = "")
          event = paste("c('", node, "')", sep="")
          evidence = constructEvidence(act, typ, chr)
          probs[[idx]] <- getinvProb(node, ev, event, evidence)
          idx<- idx+1
        }
      }
    }
  }
  return(probs)
}


constructEvidence <- function(act, typ, char){
  st<- paste("list(AACT=", "'", act, "'", " , ", "AT=", "'",typ, "'", " , ", "AC=", "'",char, "')", sep="")
  return(st)
}

constructRCTEvidence <- function(act, typ, char){
  st<- paste("list(RRCT=", "'", act, "'", " , ", "AT=", "'",typ, "'", " , ", "AC=", "'",char, "')", sep="")
  return(st)
}

eventValues <- c("LOW", "HIGH")
evidence1Values <- c("Attack", "Taunt", "Walk")
evidence2Values <- c("Type1", "Type2", "Type3")
evidence3Values <- c("Satyr", "Golem")

RcteventValues <- c("LOW", "HIGH")
Rctevidence1Values <- c("Dying", "Hurt", "Idle", "Attack")
Rctevidence2Values <- c("Type1", "Type2", "Type3")
Rctevidence3Values <- c("Satyr", "Golem")


getRctCptNode <- function(node){
  idx <- 1
  probs <- c()
  for(act in Rctevidence1Values){
    for(typ in Rctevidence2Values){
      for(chr in Rctevidence3Values){
        for(ev in RcteventValues){
          #event = paste("(", node, "==", "'", ev, "')", sep = "")
          event = paste("c('", node, "')", sep="")
          evidence = constructRCTEvidence(act, typ, chr)
          probs[[idx]] <- getinvProb(node, ev, event, evidence)
          idx<- idx+1
        }
      }
    }
  }
  return(probs)
}


# Actor nodes will be indexed by Action, Type and character

act_probs <- getActionInvCpt()
rct_probs <- getReactionInvCpt()

game_net_full <- model2network("[Actor Character][Reactor Character][Actor Type|Actor Character][Reactor Type|Reactor Character][Actor Strength|Actor Type:Actor Character][Actor Defense|Actor Type:Actor Character][Actor Attack|Actor Type:Actor Character][Reactor Strength|Reactor Type:Reactor Character][Reactor Defense|Reactor Type:Reactor Character][Reactor Attack|Reactor Type:Reactor Character][Action|Actor Strength:Actor Defense:Actor Attack][Reaction|Reactor Strength:Reactor Defense:Reactor Attack:Action][IMG|Action:Reaction:Actor Character:Reactor Character:Reactor Type:Actor Type]") 


#game_net_full <- model2network("[Z][Actor Character][Reactor Character][Actor Type|Actor Character][Reactor Type|Reactor Character][Actor Strength|Actor Type:Actor Character][Actor Defense|Actor Type:Actor Character][Actor Attack|Actor Type:Actor Character][Reactor Strength|Reactor Type:Reactor Character][Reactor Defense|Reactor Type:Reactor Character][Reactor Attack|Reactor Type:Reactor Character][Action|Actor Strength:Actor Defense:Actor Attack][Reaction|Reactor Strength:Reactor Defense:Reactor Attack:Action][IMG|Action:Reaction:Actor Character:Reactor Character:Reactor Type:Actor Type:Z]") 
graphviz.plot(game_net_full, shape="rectangle") 




# Intervention 1 - Setting manually the attributes of the actor and the reactor.


intervention_1_bn <- mutilated(dfit, list(AA="HIGH", AD="HIGH", AS="HIGH", RD="LOW", RS="LOW", RA="LOW"))
graphviz.plot(intervention_1_bn, shape="rectangle")
intervention_1_grain <- as.grain(intervention_1_bn)
