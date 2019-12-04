module AEDSim

using Distributions, Random, Distributed, SharedArrays, Statistics, WriteVTK

"""
patients have sz intervals between sz and sztimes marking the times of the sz; these
are both to be interpreted in units of days and fractions thereof

should observe an invariant that the sz times are cumsum(intervals)
"""
struct patient
  szintvls :: Array{Float64,1}
  sztimes :: Array{Float64,1}
end

"""
return a new patient with a characteristic sz interval and with a specified
number of seizures. The sz interval is used as the single parameter for an
Exponential distribution
"""
function makePatient(szinterval, szN)
  rv = Distributions.Exponential(szinterval)
  intervals = rand(rv, szN)
  szTimes = cumsum(intervals)
  return patient(intervals, szTimes)
end

"""
return *n* patients with characteristic sz intervals assigned randomly from
a chosen distribution, each with *nSz* seizures each
"""
function makeNPatients(n, dist, nSz)
  pts :: Array{patient,1} = Array{patient}(undef, n)
  params = rand(dist, n)
  for i in 1:n
    pts[i] = makePatient(params[i], nSz)
  end
  return pts
end

goodDist = LogNormal(3, 1.5)

"""
find the mean seizure interval for a patient
"""
function meanSzInterval(p ::patient)
  return p.szintvls |> mean
end

"""
return the number of sz a patient has up until (but not including) a day
"""
function countSzUpTo(pt :: patient, uptoDay ::Int)
  return length(filter(x -> x<uptoDay,  pt.sztimes))
end

"""
return the number of sz between *startDay* including during it and up to but not including
*endDay* for a particular *pt* patient.
"""
function ptSzBetweenDays(pt :: patient, startDay :: Int, endDay :: Int)
  length(filter(x->x>=startDay && x<endDay, pt.sztimes))
end

"""
return whether a patient qualifies during the baseline period for a study
based on whether they had enough seizures or not, optionally starting at a given
day for the patient
"""
function qualifyBaseline(pt :: patient, baselineLength, qSzN, ptday=0)
  ptSzBetweenDays(pt, ptday, baselineLength+ptday)>=qSzN
end

"""
filter a panel of patients by whether they qualify in the baseline period
"""
function whichQualifyingPts(pts :: Array{patient,1}, baselineLength, qSzN, ptday=0)
  filter(pt->qualifyBaseline(pt,baselineLength,qSzN,ptday), pts)
end




"""
run an experiment with basic params
"""
function runExperiment(baseTime, obsTime, szQualperWk, ptN = 10000, szN = 100, dist=goodDist, sumfun=median) :: Union{Float64, Nothing}
  pts = makeNPatients(ptN, dist, szN) 
  baseline = baseTime
  obs = obsTime
  ptsQ = whichQualifyingPts(pts, baseline, szQualperWk*baseline)
  szBase = [ptSzBetweenDays(ptI, 0, baseline)/baseline for ptI in ptsQ]
  szObs = [ptSzBetweenDays(ptI, baseline, baseline+obs)/obs for ptI in ptsQ]
  ptsQn = length(ptsQ)
  szChg = zeros(ptsQn)
  
  for pt in 1:ptsQn
    szChg[pt] = (szObs[pt]/obs) / (szBase[pt]/baseline)
  end #for

  return !isempty(szChg) ? sumfun(szChg) : 1.0

end #function


function runPExp(baseRange, obsRange, szQualRange, distParamRange, sumfun=median) :: SharedArray{Float64}
  v = SharedArray{Float64}(baseRange, obsRange, szQualRange, distParamRange)
  tt = @distributed for i = 1:baseRange
    for j = 1:obsRange
      for k = 1:szQualRange
        for l = 1:distParamRange
          f = runExperiment(i*7, j*7, k, 10000, 10000, LogNormal(l,1.5))
          v[i,j,k,l] = f == nothing ? -1.0 : f
        end
      end
    end
  end
  wait(tt)
  return v
end

#given a filename and a 4D data array output by the experiments above, will
#write out a .vtr VTK file
function writeout(filename, data)
  (dima,dimb,dimc,dimd) = size(data)
  vtkfile = vtk_grid(filename, collect(1:dima), collect(1:dimb), collect(1:dimc))
  for i in 1:dimd
    vtk_point_data(vtkfile, data[:,:,:1], "param$i")
  end
  outfile = vtk_save(vtkfile)
end #function

end #module
