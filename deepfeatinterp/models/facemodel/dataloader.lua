--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  Multi-threaded data loader
--

local datasets = require 'datasets/init'
local Threads = require 'threads'
Threads.serialization('threads.sharedserialize')

local M = {}
local DataLoader = torch.class('resnet.DataLoader', M)

function DataLoader.create(opt, split)
  if split then
    -- Load specific split
    local dataset = datasets.create(opt, split)
    return M.DataLoader(dataset, opt, split)
  else
    -- Load train, val and test
    local loaders = {}
    for i, split in ipairs{'train', 'val', 'test'} do
      local dataset = datasets.create(opt, split)
      loaders[i] = M.DataLoader(dataset, opt, split)
    end
    return table.unpack(loaders)
  end
end

function DataLoader:__init(dataset, opt, split)
  local manualSeed = opt.manualSeed
  local function init()
    require('datasets/' .. opt.dataset)
  end
  local function main(idx)
    if manualSeed ~= 0 then
      torch.manualSeed(manualSeed + idx)
    end
    torch.setnumthreads(1)
    _G.dataset = dataset
    _G.preprocess = dataset:preprocess()
    return dataset:size()
  end

  local threads, sizes = Threads(opt.nThreads, init, main)
  self.nCrops = ((split == 'val' or split == 'test') and opt.tenCrop) and 10 or 1
  self.threads = threads
  self.batchSize = math.floor(opt.batchSize / self.nCrops)

  local size = (split == 'train') and math.min(sizes[1][1], opt.nTrain) or sizes[1][1]
  self.sampleIndices = torch.sort(torch.randperm(sizes[1][1]):narrow(1, 1, size))
  self.__size = size
  if split == 'train' then
    print(size)
  end
end

function DataLoader:size()
  return math.ceil(self.__size / self.batchSize)
end

function DataLoader:run()
  local threads = self.threads
  local size, batchSize = self.sampleIndices:size(1), self.batchSize
  local perm = self.sampleIndices:index(1, torch.randperm(size):long())

  local idx, sample = 1, nil
  local function enqueue()
    while idx <= size and threads:acceptsjob() do
      local indices = perm:narrow(1, idx, math.min(batchSize, size - idx + 1))
      threads:addjob(
        function(indices, nCrops)
          local sz = indices:size(1)
          local batch, imageSize
          local target = torch.IntTensor(sz)
          local filenames = {}
          for i, idx in ipairs(indices:totable()) do
            local sample = _G.dataset:get(idx)
            print('before preprocess',sample.input:size())
            local input = _G.preprocess(sample.input)
            if not batch then
              imageSize = input:size():totable()
              if nCrops > 1 then table.remove(imageSize, 1) end
              batch = torch.FloatTensor(sz, nCrops, table.unpack(imageSize))
            end
            batch[i]:copy(input)
            target[i] = sample.target
            table.insert(filenames, sample.filename)
          end
          collectgarbage()
          return {
            input = batch:view(sz * nCrops, table.unpack(imageSize)),
            target = target,
            filename = filenames,
          }
        end,
        function(_sample_)
          sample = _sample_
        end,
        indices,
        self.nCrops
      )
      idx = idx + batchSize
    end
  end

  local n = 0
  local function loop()
    enqueue()
    if not threads:hasjob() then
      return nil
    end
    threads:dojob()
    if threads:haserror() then
      threads:synchronize()
    end
    enqueue()
    n = n + 1
    return n, sample
  end

  return loop
end

return M.DataLoader
