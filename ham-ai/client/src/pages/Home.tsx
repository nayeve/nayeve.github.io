import Header from "@/components/Header";
import Footer from "@/components/Footer";
import SecurityBanner from "@/components/SecurityBanner";
import MatrixTable from "@/components/MatrixTable";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Shield, Radio, Signal, Zap, Users, AlertTriangle } from "lucide-react";

export default function Home() {
  return (
    <div className="min-h-screen bg-white">
      <Header />
      <SecurityBanner />
      
      {/* Hero Section */}
      <section className="bg-gradient-to-r from-slate-800 to-slate-700 text-white py-16">
        <div className="max-w-7xl mx-auto px-4">
          <div className="text-center">
            <h1 className="text-4xl md:text-6xl font-bold mb-6">
              AI Applications in Amateur Radio
            </h1>
            <p className="text-xl md:text-2xl mb-8 opacity-90">
              Comprehensive Matrix of Practical, Replicable AI Applications
            </p>
            <div className="bg-white/10 backdrop-blur-sm rounded-lg p-6 inline-block">
              <p className="text-lg mb-2">
                <strong>Authored by:</strong> KD8MWZ
              </p>
              <p className="text-sm opacity-80">
                Licensed Amateur Radio Operator • AI Research Specialist
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Introduction Section */}
      <section className="py-12 bg-white">
        <div className="max-w-7xl mx-auto px-4">
          <div className="max-w-4xl mx-auto">
            <h2 className="text-3xl font-bold text-gov-navy mb-6 text-center">
              Executive Summary
            </h2>
            <div className="prose prose-lg mx-auto text-gray-700">
              <p className="text-lg leading-relaxed mb-6">
                This comprehensive matrix presents practical artificial intelligence applications currently deployed 
                or under development within the amateur radio community. Each application has been verified through 
                authoritative sources and represents replicable implementations suitable for amateur radio operators 
                of varying technical expertise levels.
              </p>
              <p className="text-lg leading-relaxed mb-6">
                The research consolidates findings from multiple authoritative sources including technical publications, 
                open-source projects, and commercial implementations to provide operators with actionable intelligence 
                on AI integration opportunities within their amateur radio stations.
              </p>
              <p className="text-lg leading-relaxed mb-6">
                Examples are drawn from both commercial/experimental tools and open-source projects, including RM Noise, 
                RadioML, WSJT-X, fldigi, Whisper, and LLMs like ChatGPT and Claude.ai. Each entry in the matrix includes 
                detailed implementation guides, hardware/software requirements, and step-by-step setup instructions to 
                enable practical deployment in amateur radio stations.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Applications Matrix Section */}
      <section id="applications" className="py-12 gov-bg">
        <div className="max-w-7xl mx-auto px-4">
          <h2 className="text-3xl font-bold text-gov-navy mb-8 text-center">
            Matrix of AI Applications in Amateur Radio
          </h2>
          
          <MatrixTable />

          {/* Additional Applications Summary */}
          <div className="mt-8 grid md:grid-cols-3 gap-6">
            <Card className="p-6">
              <CardContent className="p-0">
                <h3 className="text-lg font-semibold text-gov-navy mb-3 flex items-center">
                  <Signal className="mr-2 h-5 w-5" />
                  Signal Processing (15 Applications)
                </h3>
                <p className="text-gray-600 text-sm">
                  Advanced AI algorithms for noise reduction, signal detection, and mode recognition
                </p>
              </CardContent>
            </Card>
            <Card className="p-6">
              <CardContent className="p-0">
                <h3 className="text-lg font-semibold text-gov-navy mb-3 flex items-center">
                  <Radio className="mr-2 h-5 w-5" />
                  Station Operations (12 Applications)
                </h3>
                <p className="text-gray-600 text-sm">
                  Automated logging, contest support, and intelligent station management
                </p>
              </CardContent>
            </Card>
            <Card className="p-6">
              <CardContent className="p-0">
                <h3 className="text-lg font-semibold text-gov-navy mb-3 flex items-center">
                  <AlertTriangle className="mr-2 h-5 w-5" />
                  Emergency Communications (8 Applications)
                </h3>
                <p className="text-gray-600 text-sm">
                  Critical AI tools for emergency response and public service communications
                </p>
              </CardContent>
            </Card>
          </div>
        </div>
      </section>

      {/* Research Sources Section */}
      <section className="py-12 bg-white">
        <div className="max-w-7xl mx-auto px-4">
          <h2 className="text-3xl font-bold text-gov-navy mb-8 text-center">
            Authoritative Sources & Citations
          </h2>
          
          <div className="max-w-4xl mx-auto">
            <Card className="p-8">
              <CardContent className="p-0">
                <h3 className="text-xl font-semibold text-gov-navy mb-6">Research Methodology</h3>
                <p className="text-gray-700 mb-6">
                  This matrix represents the current landscape of practical, replicable AI applications in amateur radio, 
                  verified against authoritative sources and real-world deployments. Examples are drawn from both 
                  commercial/experimental tools and open-source projects.
                </p>
                
                <div className="grid md:grid-cols-2 gap-6 mt-8">
                  <div>
                    <h4 className="font-semibold text-gov-navy mb-3">Primary Sources</h4>
                    <div className="space-y-2 text-sm">
                      <div className="flex items-start">
                        <span className="text-gov-blue font-mono mr-2">[^1]</span>
                        <a href="https://brokensignal.tv/pages/AI_in_your_Ham_Shack.html" 
                           className="text-gov-blue hover:underline" target="_blank" rel="noopener noreferrer">
                          BrokenSignal.tv - AI in Amateur Radio
                        </a>
                      </div>
                      <div className="flex items-start">
                        <span className="text-gov-blue font-mono mr-2">[^2]</span>
                        <a href="https://www.iaru-r1.org/2025/iaru-innovation-zone-ham-radio-2025/" 
                           className="text-gov-blue hover:underline" target="_blank" rel="noopener noreferrer">
                          IARU Innovation Zone
                        </a>
                      </div>
                      <div className="flex items-start">
                        <span className="text-gov-blue font-mono mr-2">[^5]</span>
                        <a href="http://ykars.com/index.php/amateur-radio/technical-notes/ai-and-amateur-radio" 
                           className="text-gov-blue hover:underline" target="_blank" rel="noopener noreferrer">
                          YKARS Technical Notes
                        </a>
                      </div>
                      <div className="flex items-start">
                        <span className="text-gov-blue font-mono mr-2">[^6]</span>
                        <a href="https://www.linkedin.com/pulse/future-ai-amateur-radio-revolutionizing-hobby-samir-khayat-ppogf" 
                           className="text-gov-blue hover:underline" target="_blank" rel="noopener noreferrer">
                          LinkedIn Research by Samir Khayat
                        </a>
                      </div>
                    </div>
                  </div>
                  <div>
                    <h4 className="font-semibold text-gov-navy mb-3">Supporting Sources</h4>
                    <div className="space-y-2 text-sm">
                      <div className="flex items-start">
                        <span className="text-gov-blue font-mono mr-2">[^7]</span>
                        <a href="https://www.openresearch.institute/2025/03/21/inner-circle-newsletter-february-2025/" 
                           className="text-gov-blue hover:underline" target="_blank" rel="noopener noreferrer">
                          Open Research Institute
                        </a>
                      </div>
                      <div className="flex items-start">
                        <span className="text-gov-blue font-mono mr-2">[^8]</span>
                        <a href="https://sourceforge.net/directory/ham-radio/" 
                           className="text-gov-blue hover:underline" target="_blank" rel="noopener noreferrer">
                          SourceForge Ham Radio Projects
                        </a>
                      </div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </section>

      {/* Author Section */}
      <section className="py-12 gov-bg">
        <div className="max-w-7xl mx-auto px-4">
          <div className="max-w-3xl mx-auto text-center">
            <h2 className="text-3xl font-bold text-gov-navy mb-6">About the Author</h2>
            <Card className="p-8">
              <CardContent className="p-0">
                <div className="mb-6">
                  <div className="w-24 h-24 bg-slate-800 rounded-full flex items-center justify-center mx-auto mb-4">
                    <Radio className="text-white h-12 w-12" />
                  </div>
                  <h3 className="text-2xl font-bold text-gov-navy">KD8MWZ</h3>
                  <p className="text-gov-gray">Licensed Amateur Radio Operator</p>
                </div>
                <div className="text-left space-y-4 text-gray-700">
                  <p>
                    <strong>Technical Expertise:</strong> Specializing in artificial intelligence applications 
                    within amateur radio communications, with focus on practical implementations and 
                    real-world deployment strategies.
                  </p>
                  <p>
                    <strong>Research Focus:</strong> Integration of machine learning algorithms with 
                    traditional amateur radio equipment and practices, emphasizing accessibility and 
                    replicability for operators of all skill levels.
                  </p>
                  <p>
                    <strong>Professional Background:</strong> Contributing to the advancement of amateur radio 
                    technology through research, documentation, and community education on emerging AI applications.
                  </p>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </section>

      <Footer />
    </div>
  );
}
