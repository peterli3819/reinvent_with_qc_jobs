import json
import os
import shutil
import subprocess
from sys import stdout
import tempfile
import time
import django
import datetime
import copy

from reinvent_scoring.scoring.score_summary import ComponentSummary

# Right here, change to match your database
os.environ["DJANGO_SETTINGS_MODULE"] = "djangochem.settings.default"
# this must be run to setup access to the django settings and make database access work etc.
django.setup()
# import the models that you want to access
from pgmols.models import Calc
from jobs.models import Job

from django.core.management import call_command

import numpy as np
from typing import List, Tuple

from reinvent_scoring.scoring.utils import _is_development_environment

from reinvent_scoring.scoring.component_parameters import ComponentParameters
from reinvent_scoring.scoring.score_components.console_invoked.base_console_invoked_component import BaseConsoleInvokedComponent


class ExJobs(BaseConsoleInvokedComponent):
    def __init__(self, parameters: ComponentParameters):
        super().__init__(parameters)
        #self._executor_path = self.parameters.specific_parameters[self.component_specific_parameters.ICOLOS_EXECUTOR_PATH]
        #self._configuration_path = self.parameters.specific_parameters[self.component_specific_parameters.ICOLOS_CONFPATH]
        #self._values_key = self.parameters.specific_parameters[self.component_specific_parameters.ICOLOS_VALUES_KEY]

        self._project_name = self.parameters.specific_parameters[self.component_specific_parameters.JOB_PROJECT_NAME]
        self._tag = self.parameters.specific_parameters[self.component_specific_parameters.JOB_TAG]
        self._chemconfig = self.parameters.specific_parameters[self.component_specific_parameters.JOB_CHEMCONFIG]
        self._job_name = self.parameters.specific_parameters[self.component_specific_parameters.JOB_JOB_NAME]
        self._jobbuild_dir = self.parameters.specific_parameters[self.component_specific_parameters.JOB_JOBBUILD_DIR]
        self._jobparse_dir = self.parameters.specific_parameters[self.component_specific_parameters.JOB_JOBPARSE_DIR]

    def _create_command(self, step, input_json_path: str, output_json_path: str):
        # use "step" as well for the write-out
        global_variables = "".join(["\"input_json_path:", input_json_path, "\" ",
                                    "\"output_json_path:", output_json_path, "\" ",
                                    "\"step_id:", str(step), "\""])
        command = ' '.join([self._executor_path,
                            "-conf", self._configuration_path,
                            "--global_variables", global_variables])

        # check, if Icolos is to be executed in debug mode, which will cause its loggers to print out
        # much more detailed information
        command = self._add_debug_mode_if_selected(command)
        return command

    def _prepare_input_data_JSON(self, path: str, smiles: List[str]):
        """Needs to look something like:
           {
               "names": ["0", "1", "3"],
               "smiles": ["C#CCCCn1...", "CCCCn1c...", "CC(C)(C)CCC1(c2..."]
           }"""
        names = [str(idx) for idx in range(len(smiles))]
        input_dict = {"names": names,
                      "smiles": smiles}
        with open(path, 'w') as f:
            json.dump(input_dict, f, indent=4)

    def _select_values(self, data: dict) -> list:
        for value_dict in data["results"]:
            if self._values_key == value_dict[self.component_specific_parameters.ICOLOS_VALUES_KEY]:
                return value_dict["values"]
        return []

    def _parse_output_data_json(self, path: str) -> Tuple[List[str], List[float]]:
        """Needs to look something like:
           {
               "results": [{
                   "values_key": "docking_score",
                   "values": ["-5.88841", "-5.72676", "-7.30167"]},
                           {
                   "values_key": "shape_similarity",
                   "values": ["0.476677", "0.458017", "0.510676"]},
                           {
                   "values_key": "esp_similarity",
                   "values": ["0.107989", "0.119446", "0.100109"]}],
               "names": ["0", "1", "2"]
           }"""
        names_list = []
        values_list = []

        if not os.path.isfile(path):
            raise FileNotFoundError(f"Output file {path} does not exist, indicating that execution of Icolos failed entirely. Check your setup and the log file.")

        with open(path, 'r') as f:
            data = f.read().replace("\r", "").replace("\n", "")
        data = json.loads(data)
        raw_values_list = self._select_values(data=data)

        for idx in range(len(data["names"])):
            names_list.append(data["names"][idx])
            try:
                score = float(raw_values_list[idx])
            except ValueError:
                score = 0
            values_list.append(score)

        return names_list, values_list

    def _execute_command(self, command: str, final_file_path: str = None):
        # execute the pre-defined command
        result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=True)
        # TODO: once logging is available here, check the return code
        # if result.returncode != 0:

        # wait in case the final file has not been written yet or is empty (in case of a filesystem delay / hick-up)
        if final_file_path is not None:
            for _ in range(5):
                if os.path.isfile(final_file_path) and os.path.getsize(final_file_path) > 0:
                    break
                else:
                    time.sleep(3)


    def _addsmiles(self, smiles: List[str]):
        # adds the SMILES to the database
        with open("smiles.txt", "w") as f:
            f.write("\n".join(smiles))
        with open("smiles.log", "a") as f:
            f.write("adding smiles......\n")
            f.write("\n".join(smiles))
            f.write("\n")
        infile = "smiles.txt"
        
        with open("addsmiles.txt", "a") as f:
            f.write("start to add smiles\n")
            call_command("addsmiles", self._project_name, infile, tag=[self._tag])
        
    def _requestjobs(self):
        # request jobs from the database
        with open("requestjobs.txt", "a") as f:
            f.write("start to request jobs\n")
            call_command('requestjobs', self._project_name, self._chemconfig, tag=[self._tag], stdout=f)

        with open("requestjobs.txt", "r") as f:   
            output = f.readlines()

        req_time = datetime.datetime.now()
        
        return req_time # return the time when the jobs are requested
        #return int(output[-1].split()[1]), req_time # return number of requested jobs

    def _buildjobs(self):
        # request jobs from the database
        with open("buildjobs.txt", "a") as f:
            f.write("start to build jobs\n")
            call_command('buildjobs', self._project_name, self._jobbuild_dir, config=self._chemconfig, batchsize=500, stdout=f)
        
        with open("buildjobs.txt", "r") as f:   
            output = f.readlines()
        
        #return int(output[-1].split()[-1]) # return number of built jobs

    def _runjobs(self):
        # run the jobs
        cwd = os.getcwd()
        os.chdir(self._jobbuild_dir)
        jobid_list = list()
        command = f"squeue -u $USER | grep {self._job_name} | awk '{{print $1}}'"
        init_set = set(subprocess.check_output(command, shell=True).decode().split())
        for iter in os.scandir('./'):
            if iter.is_dir():
                os.chdir(iter.path)
                os.system('sbatch job_grace.sh')
                #time.sleep(30)
                #jobid = os.system("squeue -u $USER | tail -1 | awk '{print $1}'")
                tmp_set = copy.deepcopy(init_set)
                while not (tmp_set - init_set):
                    tmp_set = set(subprocess.check_output(command, shell=True).decode().split())
                    #jobid = os.system("squeue -u $USER | tail -1 | awk '{print $1}'")                  
                    try:
                        jobid = list(tmp_set - init_set)[0]
                        jobid_list.append(jobid)
                    except:
                        pass
                    time.sleep(1)
                os.chdir('..')
        #print(jobid_list)
        os.chdir(cwd)

        tmp_list = copy.deepcopy(jobid_list)
        while len(tmp_list) > 0:
            for idx in range(len(jobid_list)):
                jobid = jobid_list[idx]
                # check if exit code is 0, if not, job is finished and should be removed from the list
                if os.system(f"squeue -u $USER | grep {jobid}") != 0: 
                    try:
                        tmp_list.remove(jobid)
                    except:
                        pass

            time.sleep(60)
        
    def _parsejobs(self, smiles: List[str], req_time):
        # parse the jobs
        with open("parsejobs.txt", "a") as f:
            call_command('parsejobs', self._project_name, self._jobbuild_dir, root_path=self._jobparse_dir, stdout=f)

        name_list = [idx for idx in range(len(smiles))]
        value_list = list()

        job = Job.objects.filter(group__name=self._project_name, status='done', config__name=self._chemconfig, createtime__lte=req_time)
        for smi in smiles:
            calc = Calc.objects.filter(mol__smiles=smi, mol__tags__contains=[self._tag], parentjob__in=job)
            if len(calc) != 0:
                ex_states = calc[0].props['excitedstates']
                s1_energy = min([state['energy'] for state in ex_states if state['multiplicity'] == 'singlet'])
                t1_energy = min([state['energy'] for state in ex_states if state['multiplicity'] == 'triplet'])
                s1_calibrated = 0.72690296 * s1_energy + 0.5986109788327894
                t1_calibrated = 0.79066724 * t1_energy + 0.2739285134008771
                s1_t1_gap = s1_calibrated - 2 * t1_calibrated
                print(smi, s1_t1_gap)
                value_list.append(s1_t1_gap)
            else:
                value_list.append(None)

        return name_list, value_list

    def _calculate_score(self, smiles: List[str], step) -> np.array:
        # make temporary folder and set input and output paths
        #tmp_dir = tempfile.mkdtemp()
        #input_json_path = os.path.join(tmp_dir, "input.json")
        #output_json_path = os.path.join(tmp_dir, "output.json")

        # add the SMILES to the database
        self._addsmiles(smiles)

        # request jobs from the database
        req_time = self._requestjobs()

        # build the jobs
        self._buildjobs()
        #assert num_jobs == num_built, "Number of requested jobs does not match number of built jobs"

        # run the jobs
        self._runjobs()

        # parse the jobs
        smiles_ids, scores = self._parsejobs(smiles=smiles, req_time=req_time)

        # apply transformation
        transform_params = self.parameters.specific_parameters.get(
            self.component_specific_parameters.TRANSFORMATION, {}
        )
        transformed_scores = self._transformation_function(scores, transform_params)

        # clean up
        #if os.path.isdir(tmp_dir):
        #    shutil.rmtree(tmp_dir)

        return np.array(transformed_scores), np.array(scores)
